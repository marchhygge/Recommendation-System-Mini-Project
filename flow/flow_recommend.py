from prefect import flow, task
from recommend import recommendation_v1 as rec

@task
def task_load_env():
    env = rec.load_env_variables()
    return env

@task
def task_connect_db(env):
    con, cursor = rec.connect_db(env)
    return con, cursor

@task
def task_fetch_data(cursor):
    df_restaurants, df_users = rec.fetch_datas(cursor)
    return df_restaurants, df_users

@task
def task_preprocess(df_restaurants, df_users):
    restaurant_profiles, user_profiles = rec.preprocess_data(df_restaurants, df_users)
    return restaurant_profiles, user_profiles

@task
def task_compute_similarity(restaurant_profiles, user_profiles):
    df_recommendations = rec.compute_similarity(restaurant_profiles, user_profiles)
    return df_recommendations

@task
def task_write_DB(con, cursor, df_recommendations):
    rec.write_DB(con, cursor, df_recommendations)

@task
def task_create_view(cursor, con):
    rec.create_view(cursor, con)

@flow(name="Recommendation Flow")
def recommend_pipeline():
    print("Starting {name}")

    # Load env
    env = task_load_env()

    # Connect DB
    con, cursor = task_connect_db(env)

    # Fetch data
    df_restaurants, df_users = task_fetch_data(cursor)

    # Preprocess
    restaurant_profiles, user_profiles = task_preprocess(df_restaurants, df_users)

    # Compute similarity
    df_recommendations = task_compute_similarity(restaurant_profiles,user_profiles)

    # Write DB
    task_write_DB(con, cursor, df_recommendations)

    # Create view
    task_create_view(cursor, con)

    print("{name} completed successfully.")

if __name__ == "__main__":
    recommend_pipeline()