import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from sklearn.metrics import mean_absolute_error


# -----------------------
# Athena Query Function
# -----------------------
@st.cache_data
def run_athena_query(hour, query: str, database: str, s3_output: str) -> pd.DataFrame:
    athena_client = boto3.client('athena', region_name="us-east-1")

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': s3_output}
    )

    query_execution_id = response['QueryExecutionId']
    state = 'RUNNING'

    while state in ['RUNNING', 'QUEUED']:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        if state in ['RUNNING', 'QUEUED']:
            time.sleep(1)

    if state != 'SUCCEEDED':
        reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
        raise Exception(f"Athena query failed: {state} - {reason}")

    results = []
    columns = []
    next_token = None
    first_page = True

    while True:
        if next_token:
            result_set = athena_client.get_query_results(
                QueryExecutionId=query_execution_id,
                NextToken=next_token
            )
        else:
            result_set = athena_client.get_query_results(QueryExecutionId=query_execution_id)

        if first_page:
            columns = [col['Label'] for col in result_set['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            first_page = False

        rows = result_set['ResultSet']['Rows']
        if next_token is None:
            rows = rows[1:]  # skip header

        for row in rows:
            results.append([field.get('VarCharValue', '') for field in row['Data']])

        next_token = result_set.get('NextToken')
        if not next_token:
            break

    df = pd.DataFrame(results, columns=columns)

    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff[denominator != 0]) * 100


# -----------------------
# Streamlit App
# -----------------------
st.title("NYC Taxi Rides Forecast")

# Tabs
tab1, tab2 = st.tabs(["Athena", "RDS"])

# -----------------------
# Tab: Athena
# -----------------------
with tab1:
    # Pickup Location Input
    location_map = {
        "Central Park, Manhattan": 43,
        "LaGuardia Airport, Queens": 138,
        "Upper East Side North, Manhattan": 237
    }
    
    # Dropdown input
    selected_location = st.selectbox("Select Pickup Location", options=list(location_map.keys()))
    
    # Get the corresponding ID
    location_id = location_map[selected_location]

    # Use Eastern Time (New York)
    eastern = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=eastern)

    # Calculate the same week last year in NY time
    end_date = now_ny - timedelta(days=358, hours = 1)
    start_date = now_ny - timedelta(days=365)
    
    # Round down to start of the hour
    start_rounded = start_date.replace(minute=0, second=0, microsecond=0)

    # Round up to next full hour if needed
    if end_date.minute > 0 or end_date.second > 0 or end_date.microsecond > 0:
        end_rounded = (end_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        end_rounded = end_date.replace(minute=0, second=0, microsecond=0)

    next_hour = end_date + timedelta(hours=1)
    
    # Format for Athena query (still using naive-looking string)
    start_str = start_rounded.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_rounded.strftime("%Y-%m-%d %H:%M:%S")
    next_hour_str = next_hour.strftime("%Y-%m-%d %H:%M:%S")

    # Athena setup
    s3_output = 's3://s3_bucket_name/athena/'

    # Queries
    actual_query = f"""
    SELECT DISTINCT
        pickup_hour,
        rides
    FROM glue_transformed
    WHERE
        pickup_location_id = {location_id}
        AND pickup_hour BETWEEN '{start_str}' AND '{end_str}'
    ORDER BY pickup_hour;
    """

    predicted_query = f"""
    SELECT DISTINCT
        prediction_datetime,
        predicted_rides
    FROM predictions
    WHERE
        pickup_location_id = '{location_id}'
        AND model = '1'
        AND prediction_datetime BETWEEN '{start_str}' AND '{next_hour_str}'
    ORDER BY prediction_datetime;
    """

    predicted_query1 = f"""
    SELECT DISTINCT
        prediction_datetime,
        predicted_rides
    FROM predictions
    WHERE
        pickup_location_id = '{location_id}'
        AND model = '2'
        AND prediction_datetime BETWEEN '{start_str}' AND '{next_hour}'
    ORDER BY prediction_datetime;
    """

    # Run Queries
    try:
        actual_df = run_athena_query(now_ny.hour, actual_query, 'etl_taxi_transformed', s3_output)
        predicted_df = run_athena_query(now_ny.hour, predicted_query, 'etl_taxi_transformed', s3_output)
        predicted_df2 = run_athena_query(now_ny.hour, predicted_query1, 'etl_taxi_transformed', s3_output)

        # Merge and calculate metrics for both models
        metrics_data = []
        
        # Model 1
        merged1 = pd.merge(actual_df, predicted_df, left_on='pickup_hour', right_on='prediction_datetime',how='inner')
        if not merged1.empty:
            mae1 = mean_absolute_error(merged1['rides'], merged1['predicted_rides'])
            smape1 = smape(merged1['rides'], merged1['predicted_rides'])
            metrics_data.append({
                'Model': 'LightGBM (28-day Lag)',
                'MAE': f"{mae1:.2f}",
                'SMAPE (%)': f"{smape1:.2f}"
            })
        
        # Model 2
        merged2 = pd.merge(actual_df, predicted_df2, left_on='pickup_hour', right_on='prediction_datetime', how='inner')
        if not merged2.empty:
            mae2 = mean_absolute_error(merged2['rides'], merged2['predicted_rides'])
            smape2 = smape(merged2['rides'], merged2['predicted_rides'])
            metrics_data.append({
                'Model': 'LightGBM (Feature Selection)',
                'MAE': f"{mae2:.2f}",
                'SMAPE (%)': f"{smape2:.2f}"
            })


        # Plot
        # fig = px.line()
    #     fig.add_scatter(x=actual_df['pickup_hour'], y=actual_df['rides'], name='Actual Rides')
    #     fig.add_scatter(x=predicted_df['prediction_datetime'], y=predicted_df['predicted_rides'], name='Predicted Rides')
    #     fig.update_layout(title=f"Taxi Rides Forecast for Location {location_id}",
    #                       xaxis_title="Time",
    #                       yaxis_title="Number of Rides")

    #     st.plotly_chart(fig)

    # except Exception as e:
    #     st.error(f"❌ Error fetching data: {e}")


        fig = px.line()

        fig.add_scatter(x=actual_df['pickup_hour'], y=actual_df['rides'], name='Actual Rides')
        fig.add_scatter(x=predicted_df['prediction_datetime'], y=predicted_df['predicted_rides'], 
                        name='Predicted Rides (Model 1)', line=dict(color='orange'))
        fig.update_layout(title=f"Model 1: Taxi Rides Forecast for Location - {selected_location}",
                          xaxis_title="Time",
                          yaxis_title="Number of Rides")

        st.plotly_chart(fig)

        fig2 = px.line()

        fig2.add_scatter(x=actual_df['pickup_hour'], y=actual_df['rides'], name='Actual Rides')
        fig2.add_scatter(x=predicted_df2['prediction_datetime'], y=predicted_df2['predicted_rides'], 
                        name='Predicted Rides (Model 2)', line=dict(color='green'))

        fig2.update_layout(title=f"Model 2: Taxi Rides Forecast for Location - {selected_location}",
                          xaxis_title="Time",
                          yaxis_title="Number of Rides")

        st.plotly_chart(fig2)

        # Show comparison table
        if metrics_data:
            st.markdown("**Forecast Metrics**")
            st.dataframe(pd.DataFrame(metrics_data))
        else:
            st.warning("No overlapping timestamps to calculate error metrics for either model.")

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")


# -----------------------
# Tab: RDS
# -----------------------
with tab2:
    st.subheader("RDS: Actual vs Predicted Rides")

    # Pickup Location Input
    location_map = {
        "Central Park, Manhattan": 43,
        "LaGuardia Airport, Queens": 138,
        "Upper East Side North, Manhattan": 237
    }
    
    # Dropdown input
    selected_location_rds = st.selectbox("Select Pickup Location (RDS)", options=list(location_map.keys()))
    
    # Get the corresponding ID
    location_id_rds = location_map[selected_location_rds]

    # RDS connection settings
    import sqlalchemy
    from sqlalchemy import create_engine, text

    rds_host = "rds_host_name"
    rds_db = "postgres"
    rds_user = "taxiuser"
    rds_password = "ur_password_here"
    rds_port = 5432

    # Create engine
    rds_engine = create_engine(
        f"postgresql+psycopg2://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}"
    )

    # Use same time range
    try:
        with rds_engine.connect() as conn:
            actual_sql = text(f"""
                SELECT pickup_hour, rides
                FROM taxi_rides
                WHERE pickup_location_id = :loc
                AND pickup_hour BETWEEN :start AND :end
                ORDER BY pickup_hour;
            """)
            predicted_sql = text(f"""
                SELECT prediction_datetime, predicted_rides
                FROM predicted_rides_model1
                WHERE pickup_location_id = :loc
                AND prediction_datetime BETWEEN :start AND :end
                ORDER BY prediction_datetime;
            """)

            predicted_sql1 = text(f"""
                SELECT prediction_datetime, predicted_rides
                FROM predicted_rides_model2
                WHERE pickup_location_id = :loc
                AND prediction_datetime BETWEEN :start AND :end
                ORDER BY prediction_datetime;
            """)

            actual_df_rds = pd.read_sql(actual_sql, conn, params={
                'loc': location_id_rds,
                'start': start_str,
                'end': end_str
            })

            predicted_df_rds = pd.read_sql(predicted_sql, conn, params={
                'loc': location_id_rds,
                'start': start_str,
                'end': next_hour_str
            })

            predicted_df_rds1 = pd.read_sql(predicted_sql1, conn, params={
                'loc': location_id_rds,
                'start': start_str,
                'end': next_hour_str
            })

            # Plot: Model 1
            fig_rds = px.line()
            fig_rds.add_scatter(x=actual_df_rds['pickup_hour'], y=actual_df_rds['rides'], name='Actual Rides')
            fig_rds.add_scatter(x=predicted_df_rds['prediction_datetime'], y=predicted_df_rds['predicted_rides'],
                                name='Predicted Rides(Model 1)', line=dict(color='orange'))
            fig_rds.update_layout(title=f"Model 1: Taxi Rides Forecast for Location - {selected_location_rds}",
                                  xaxis_title="Time",
                                  yaxis_title="Number of Rides")
            st.plotly_chart(fig_rds)

            # Plot: Model 2
            fig_rds1 = px.line()
            fig_rds1.add_scatter(x=actual_df_rds['pickup_hour'], y=actual_df_rds['rides'], name='Actual Rides')
    
            fig_rds1.add_scatter(x=predicted_df_rds1['prediction_datetime'], y=predicted_df_rds1['predicted_rides'],
                                name='Predicted Rides (Model 2)', line=dict(color='green'))
            fig_rds1.update_layout(title=f"Model 2: Taxi Rides Forecast for Location - {selected_location_rds}",
                                  xaxis_title="Time",
                                  yaxis_title="Number of Rides")
            st.plotly_chart(fig_rds1)

            metrics_data_rds = []
            # Model 1
            merged_rds1 = pd.merge(actual_df_rds, predicted_df_rds,left_on='pickup_hour',
                                   right_on='prediction_datetime',how='inner')
            if not merged_rds1.empty:
                mae_rds1 = mean_absolute_error(merged_rds1['rides'], merged_rds1['predicted_rides'])
                smape_rds1 = smape(merged_rds1['rides'], merged_rds1['predicted_rides'])
                metrics_data_rds.append({
                    'Model': 'LightGBM (28-day Lag)',
                    'MAE': f"{mae_rds1:.2f}",
                    'SMAPE (%)': f"{smape_rds1:.2f}"
                })
            
            # Model 2
            merged_rds2 = pd.merge(actual_df_rds, predicted_df_rds1,left_on='pickup_hour', right_on='prediction_datetime',
                                   how='inner')
            if not merged_rds2.empty:
                mae_rds2 = mean_absolute_error(merged_rds2['rides'], merged_rds2['predicted_rides'])
                smape_rds2 = smape(merged_rds2['rides'], merged_rds2['predicted_rides'])
                metrics_data_rds.append({
                    'Model': 'LightGBM (Feature Selection)',
                    'MAE': f"{mae_rds2:.2f}",
                    'SMAPE (%)': f"{smape_rds2:.2f}"
                })
            
            # Show comparison table
            if metrics_data_rds:
                st.markdown("**Forecast Metrics**")
                st.dataframe(pd.DataFrame(metrics_data_rds))
            else:
                st.warning("No overlapping timestamps to calculate error metrics for either model.")

    except Exception as e:
        st.error(f"❌ Error connecting to RDS: {e}")


    

    

