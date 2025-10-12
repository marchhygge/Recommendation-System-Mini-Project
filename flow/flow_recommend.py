from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.notifications import SlackWebhook
from datetime import timedelta
import traceback
from recommend import recommendation_v1 as rec


# ============ TASK DEFINITIONS ============

@task(
    name="Load Environment Variables",
    cache_policy=NO_CACHE,
    retries=2,
    retry_delay_seconds=5
)
def task_load_env():
    logger = get_run_logger()
    logger.info("Loading environment variables...")
    try:
        env = rec.load_env_variables()
        logger.info("Environment variables loaded successfully")
        return env
    except Exception as e:
        logger.error("Failed to load environment: %s", str(e))
        raise


@task(
    name="Fetch Data from Database",
    retries=3,
    retry_delay_seconds=10
)
def task_fetch_data(env):
    logger = get_run_logger()
    logger.info("Connecting to database and fetching data...")
    
    try:
        con, cursor = rec.connect_db(env)
        logger.info("Database connection established")
        
        df_restaurants, df_users = rec.fetch_datas(cursor)
        con.close()
        
        # Validation checks
        if df_restaurants.empty or df_users.empty:
            raise ValueError("Empty dataset retrieved from database")
        
        logger.info("Fetched %d restaurants and %d users", len(df_restaurants), len(df_users))
        logger.info("Restaurant columns: %s", list(df_restaurants.columns))
        logger.info("User columns: %s", list(df_users.columns))
        
        return df_restaurants, df_users
        
    except Exception as e:
        logger.error("Failed to fetch data: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        raise


@task(
    name="Preprocess Data"
)
def task_preprocess(df_restaurants, df_users):
    logger = get_run_logger()
    logger.info("Preprocessing data...")
    
    try:
        restaurant_profiles, user_profiles = rec.preprocess_data(df_restaurants, df_users)
        
        # Validation
        if restaurant_profiles.empty or user_profiles.empty:
            raise ValueError("Preprocessing resulted in empty profiles")
        
        logger.info("Created %d restaurant profiles and %d user profiles", len(restaurant_profiles), len(user_profiles))
        logger.info("Avg tags per restaurant: %.2f", df_restaurants.groupby('restaurant_id').size().mean())
        logger.info("Avg tags per user: %.2f", df_users.groupby('user_id').size().mean())
        
        return restaurant_profiles, user_profiles
        
    except Exception as e:
        logger.error("Preprocessing failed: %s", str(e))
        raise


@task(
    name="Compute Similarity Matrix"
)
def task_compute_similarity(restaurant_profiles, user_profiles):
    logger = get_run_logger()
    logger.info("Computing similarity matrix...")
    
    try:
        df_recommendations = rec.compute_similarity(restaurant_profiles, user_profiles)
        
        # Statistics
        logger.info("Generated %d recommendations", len(df_recommendations))
        logger.info("Score statistics:")
        logger.info("   Mean: %.4f", df_recommendations['score'].mean())
        logger.info("   Median: %.4f", df_recommendations['score'].median())
        logger.info("   Min: %.4f", df_recommendations['score'].min())
        logger.info("   Max: %.4f", df_recommendations['score'].max())
        logger.info("High quality recs (score > 0.5): %d (%.1f%%)", 
                   (df_recommendations['score'] > 0.5).sum(),
                   (df_recommendations['score'] > 0.5).sum() / len(df_recommendations) * 100)
        
        return df_recommendations
        
    except Exception as e:
        logger.error("Similarity computation failed: %s", str(e))
        raise


@task(
    name="Write to Database",
    cache_policy=NO_CACHE,
    retries=2,
    retry_delay_seconds=5
)
def task_write_DB(env, df_recommendations):
    logger = get_run_logger()
    logger.info("Writing %d recommendations to database...", len(df_recommendations))
    
    try:
        con, cursor = rec.connect_db(env)
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM recommendation")
        old_count = cursor.fetchone()[0]
        logger.info("Existing recommendations in DB: %d", old_count)
        
        rec.write_DB(con, cursor, df_recommendations)
        con.close()
        
        logger.info("Recommendations saved successfully")
        logger.info("Deleted: %d, Inserted: %d", old_count, len(df_recommendations))
        
    except Exception as e:
        logger.error("Failed to write to database: %s", str(e))
        raise


@task(
    name="Create Database View",
    cache_policy=NO_CACHE,
    retries=2,
    retry_delay_seconds=5
)
def task_create_view(env):
    logger = get_run_logger()
    logger.info("Creating recommendation view...")
    
    try:
        con, cursor = rec.connect_db(env)
        rec.create_view(cursor, con)
        logger.info("View created successfully")
        
    except Exception as e:
        logger.error("Failed to create view: %s", str(e))
        raise


# ============ FLOW DEFINITION ============

@flow(
    name="Restaurant Recommendation Pipeline",
    description="End-to-end pipeline for generating restaurant recommendations using TF-IDF and cosine similarity",
    version="2.0",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
    persist_result=True,
    result_storage=None,  
    timeout_seconds=3600,  # 1 hour timeout
    validate_parameters=True
)
def recommend_pipeline():
    logger = get_run_logger()
    
    logger.info("="*70)
    logger.info("Starting Restaurant Recommendation Pipeline")
    logger.info("="*70)
    
    try:
        # Step 1: Load environment
        env = task_load_env()
        
        # Step 2: Fetch data
        df_restaurants, df_users = task_fetch_data(env)
        
        # Step 3: Preprocess
        restaurant_profiles, user_profiles = task_preprocess(df_restaurants, df_users)
        
        # Step 4: Compute similarity
        df_recommendations = task_compute_similarity(restaurant_profiles, user_profiles)
        
        # Step 5: Write to database
        task_write_DB(env, df_recommendations)
        
        # Step 6: Create view
        task_create_view(env)
        
        logger.info("="*70)
        logger.info("Pipeline completed successfully!")
        logger.info("="*70)
        
        return {
            "status": "success",
            "restaurants_processed": len(df_restaurants),
            "users_processed": len(df_users),
            "recommendations_generated": len(df_recommendations)
        }
        
    except Exception as e:
        logger.error("="*70)
        logger.error("Pipeline failed: %s", str(e))
        logger.error("="*70)
        logger.error("Full traceback: %s", traceback.format_exc())
        raise


# ============ OPTIONAL: NOTIFICATION SETUP ============

# async def send_notification_on_failure(flow, flow_run, state):
#     """Send notification when flow fails"""
#     logger = get_run_logger()
#     try:
#         # Example: Slack notification (configure SlackWebhook block first)
#         # slack_webhook = await SlackWebhook.load("recommendation-alerts")
#         # await slack_webhook.notify(
#         #     f"Recommendation Pipeline Failed\n"
#         #     f"Flow: {flow.name}\n"
#         #     f"Run ID: {flow_run.id}\n"
#         #     f"Error: {state.message}"
#         # )
#         logger.error("Pipeline failed - notification would be sent here")
#     except Exception as e:
#         logger.error("Failed to send notification: %s", str(e))


# ============ DEPLOYMENT CONFIGURATION ============

if __name__ == "__main__":
    # For local testing
    result = recommend_pipeline()
    print(f"\nPipeline Result: {result}")
    
    # For deployment (uncomment when needed):
    # from prefect.deployments import Deployment
    # from prefect.server.schemas.schedules import CronSchedule
    # 
    # deployment = Deployment.build_from_flow(
    #     flow=recommend_pipeline,
    #     name="daily-recommendations",
    #     version="2.0",
    #     work_pool_name="default-agent-pool",
    #     work_queue_name="default",
    #     schedule=CronSchedule(cron="0 2 * * *", timezone="Asia/Ho_Chi_Minh"),  # Run at 2 AM daily
    #     tags=["production", "ml", "recommendations"],
    #     description="Daily restaurant recommendation generation",
    #     parameters={},
    # )
    # 
    # deployment.apply()