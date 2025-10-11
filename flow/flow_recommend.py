from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from recommend import recommendation_v1 as rec

@task(cache_policy=NO_CACHE)
def task_load_env():
    env = rec.load_env_variables()
    return env

@task
def task_fetch_data(env):
    con, cursor = rec.connect_db(env)
    df_restaurants, df_users = rec.fetch_datas(cursor)
    con.close()
    return df_restaurants, df_users

@task
def task_preprocess(df_restaurants, df_users):
    restaurant_profiles, user_profiles = rec.preprocess_data(df_restaurants, df_users)
    return restaurant_profiles, user_profiles

@task
def task_compute_similarity(restaurant_profiles, user_profiles):
    df_recommendations = rec.compute_similarity(restaurant_profiles, user_profiles)
    return df_recommendations

@task(cache_policy=NO_CACHE)
def task_write_DB(env, df_recommendations):
    con, cursor = rec.connect_db(env)
    rec.write_DB(con, cursor, df_recommendations)
    con.close()

@task(cache_policy=NO_CACHE)
def task_create_view(env):
    con, cursor = rec.connect_db(env)
    rec.create_view(cursor, con)
    con.close()

@flow(name="Recommendation Flow")
def recommend_pipeline():
    print("🚀 Starting Recommendation Flow...")

    # Load env
    env = task_load_env()

    # Fetch data
    df_restaurants, df_users = task_fetch_data(env)

    # Preprocess
    restaurant_profiles, user_profiles = task_preprocess(df_restaurants, df_users)

    # Compute similarity
    df_recommendations = task_compute_similarity(restaurant_profiles,user_profiles)

    # Write DB
    task_write_DB(env, df_recommendations)

    # Create view
    task_create_view(env)

    print("✅ Recommendation Flow completed successfully!")

if __name__ == "__main__":
    recommend_pipeline()