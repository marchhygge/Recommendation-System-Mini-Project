from dotenv import load_dotenv
import os
from os.path import join, dirname, abspath
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------ 1. load environment variables ------------
def load_env_variables():
    # Load .env cho local development (nếu có)
    dotenv_path = Path(__file__).parent.parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print("Loaded .env for local development")
    
    # Đọc từ environment variables (work cho cả local và cloud)
    env_vars = {
        'DB_HOST': os.getenv('DB_HOST'),
        'DB_NAME': os.getenv('DB_DATABASE'),
        'DB_USER': os.getenv('DB_USER'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD'),
        'DB_PORT': os.getenv('DB_PORT', '6543'),  # default value
    }
    
    # Validate required vars
    missing = [k for k, v in env_vars.items() if v is None]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    return env_vars
    
# ------------ 2. connect to PostgreSQL database ------------
def connect_db(env):
    try:
        con = psycopg2.connect(
            host=env["DB_HOST"],
            port=env["DB_PORT"],
            database=env["DB_NAME"],
            user=env["DB_USER"],
            password=env["DB_PASSWORD"],
            sslmode='require',
            client_encoding='UTF8'
        )
        cursor = con.cursor()
        return con, cursor
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

# ------------- 3. query restaurants --------------------------
def fetch_datas(cursor):
    try:
        # Query restaurants
        restaurant_query = """
            select b.restaurant_id,
                a.name as restaurant_name,
                a.address,
                c.category_id,
                d.name as category_name,
                b.tag_id,
                c.name as tag_name
            from restaurants a
            inner join restaurant_tags b on a.id = b.restaurant_id
            inner join tags c on c.id = b.tag_id
            inner join categories d on d.id = c.category_id
        """
        cursor.execute(restaurant_query)
        restaurant_data = cursor.fetchall()
        df_restaurants = pd.DataFrame(
            restaurant_data, columns=[desc[0] for desc in cursor.description]
        )
        
        # Query users
        users_query = """
            select b.user_id,
                a.username,
                c.category_id,
                d.name as category_name,
                b.tag_id,
                c.name as tag_name
            from users a
            inner join user_tags b on b.user_id = a.id
            inner join tags c on c.id = b.tag_id
            inner join categories d on d.id = c.category_id
        """
        cursor.execute(users_query)
        user_date = cursor.fetchall()
        df_users = pd.DataFrame(
            user_date, columns=[desc[0] for desc in cursor.description]
        )

        return df_restaurants, df_users
    except Exception as e:
        print(f"Error failed to fetch data: {e}")
        raise

# ------------- 4. preprocess data --------------------------
def preprocess_data(df_restaurants, df_users):
    try:
        # Aggregate tags for each restaurant
        restaurant_profiles = (
            df_restaurants.groupby("restaurant_id")["tag_name"]
                          .apply(lambda x: ' | '.join(x))
                          .reset_index(name="restaurantCharacteristics")
        )

        # Aggregate tags for each user
        user_profiles = (
            df_users.groupby("user_id")["tag_name"]
                    .apply(lambda x: ' | '.join(x))
                    .reset_index(name="userCharacteristics")
        )

        return restaurant_profiles, user_profiles

    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        raise

# ------------- 5. compute similarity --------------------------
def compute_similarity(restaurant_profiles, user_profiles):
    try:
        # Combine all characteristics for TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        all_characteristics = pd.concat([
            user_profiles["userCharacteristics"],
            restaurant_profiles["restaurantCharacteristics"]
        ])
        vectorizer.fit(all_characteristics)

        # Transform user and restaurant characteristics
        user_vec = vectorizer.transform(user_profiles["userCharacteristics"])
        restaurant_vec = vectorizer.transform(restaurant_profiles["restaurantCharacteristics"])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(user_vec, restaurant_vec)

        # Create a DataFrame for all user–restaurant combinations
        recommendations = []
        for user_idx, user_id in enumerate(user_profiles["user_id"]):
            for restaurant_idx, restaurant_id in enumerate(restaurant_profiles["restaurant_id"]):
                recommendations.append((
                    int(user_id),
                    int(restaurant_id),
                    float(similarity_matrix[user_idx, restaurant_idx])
                ))

        df_recommendations = pd.DataFrame(
            recommendations,
            columns=["user_id", "restaurant_id", "score"]
        )
        return df_recommendations

    except Exception as e:
        print(f"Error in compute_similarity: {e}")
        raise

# ------------- 6. write to DataBase --------------------------
def write_DB(con, cursor, df_recommendations):
    try:
        # Delete old recommendations
        user_ids = df_recommendations["user_id"].unique()
        user_ids = [int(u) for u in user_ids]
        cursor.execute(
            "DELETE FROM recommendation WHERE user_id = ANY(%s)", (list(user_ids),)
        )
        con.commit()

        # Insert new recommendations
        insert_query = """
            INSERT INTO recommendation (user_id, restaurant_id, score)
            VALUES %s
        """
        execute_values(cursor, insert_query, df_recommendations.values.tolist())
        con.commit()

    except Exception as e:
        print(f"Error in write_DB: {e}")
        con.rollback()
        raise

# ------------------ 7. create view  -----------------------
def create_view(cursor, con):
    try:
        cursor.execute(
            """
            CREATE OR REPLACE VIEW recommendation_view AS
                SELECT r.id as recommendation_id,
                    u.username as user_name,
                    res.name as restaurant_name,
                    r.score
                FROM recommendation r
                JOIN users u ON r.user_id = u.id
                JOIN restaurants res ON r.restaurant_id = res.id
            """
        )
        con.commit()
    except Exception as e:
        print(f"Error in create_view: {e}")
        con.rollback()
        raise
    finally:
        cursor.close()
        con.close()