from dotenv import load_dotenv
import os
from os import join, dirname
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ------------ 1. load enironment variables ------------
def load_env_variables():
    try:
        dotenv_path = join(dirname(__file__), '.env')
        load_dotenv(dotenv_path)
        print("[1] Environment variables loaded successfully.")
        return {
            "HOST": os.getenv("HOST"),
            "DATABASE": os.getenv("DATABASE"),
            "USER": os.getenv("USER"),
            "PASSWORD": os.getenv("PASSWORD")
        }
    except Exception as e:
        print(f"Error loading .env file: {e}")
        raise
    
# ------------ 2. connect to PostgreSQL database ------------
def connect_db(env):
    try:
        con = psycopg2.connect(
            host=env["HOST"],
            database=env["DATABASE"],
            user=env["USER"],
            password=env["PASSWORD"]
        )
        cursor = con.cursor()
        print("[2] Connected to PostgreSQL database successfully.")
        return con, cursor
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

# ------------- 3. query restaurants --------------------------
def fetch_datas(cursor):
    try:
        print("[3] fetch_datas function called.")
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
        print(f"fetched {len(df_restaurants)} restaurant records.")
        
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
        print(f"fetched {len(df_users)} user records.")

        print("[3] Data fetching completed successfully.")
        return df_restaurants, df_users
    except Exception as e:
        print(f"Error failed to fetch data: {e}")
        raise

# ------------- 4. preprocess data --------------------------
def preprocess_data(df_restaurants, df_users):
    try:
        print("[4] preprocess_data function called.")
        # Aggregate tags for each restaurant
        restaurant_profiles = (
            df_restaurants.groupby("restaurant_id")["tag_name"]
                          .apply(lambda x: ' | '.join(x))
                          .reset_index(name="restaurantCharacteristics")
        )
        print(f"Created {len(restaurant_profiles)} restaurant profiles.")

        # Aggregate tags for each user
        user_profiles = (
            df_users.groupby("user_id")["tag_name"]
                    .apply(lambda x: ' | '.join(x))
                    .reset_index(name="userCharacteristics")
        )
        print(f"Created {len(user_profiles)} user profiles.")

        print("[4] Preprocessing completed successfully.")
        return restaurant_profiles, user_profiles

    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        raise

# ------------- 5. compute similarity --------------------------
def compute_similarity(restaurant_profiles, user_profiles):
    try:
        print("[5] compute_similarity function called.")

        # Combine all characteristics for TF-IDF vectorization
        print("Combining characteristics for TF-IDF vectorization...")
        vectorizer = TfidfVectorizer()
        all_characteristics = pd.concat([
            user_profiles["userCharacteristics"],
            restaurant_profiles["restaurantCharacteristics"]
        ])
        vectorizer.fit(all_characteristics)
        print("TF-IDF vectorizer fitted.")

        # Transform user and restaurant characteristics
        print("Transforming characteristics...")
        user_vec = vectorizer.transform(user_profiles["userCharacteristics"])
        restaurant_vec = vectorizer.transform(restaurant_profiles["restaurantCharacteristics"])
        print("TF-IDF transformation completed.")

        # Compute cosine similarity
        print("Computing cosine similarity...")
        similarity_matrix = cosine_similarity(user_vec, restaurant_vec)
        print("Cosine similarity computation completed.")

        # Create a DataFrame for similarity scores
        print("Creating similarity DataFrame...")
        recommendations = []
        for user_idx, user_id in enumerate(user_profiles["user_id"]):
            top_k_idx = np.argpartition(-similarity_matrix[user_idx], k)[:k]
            top_k_sorted = top_k_idx[np.argsort(-similarity_matrix[user_idx][top_k_idx])]
            for restaurant_idx in top_k_sorted:
                recommendations.append((
                    int(user_id),
                    int(restaurant_profiles.iloc[restaurant_idx]["restaurant_id"]),
                    float(similarity_matrix[user_idx, restaurant_idx])
                ))
        df_recommendations = pd.DataFrame(recommendations, columns=["user_id","restaurant_id","score"])
        print("Similarity DataFrame created.")
        print("[5] Similarity computation completed successfully.")
        return df_recommendations
    
    except Exception as e:
        print(f"Error in compute_similarity: {e}")
        raise

# ------------- 6. write to DataBase --------------------------
def write_DB(con, cursor, df_recommendations):
    try:
        print("[6] write_DB function called.")

        # delete old recommendations
        print("deleting old recommendations...")
        user_ids = df_recommendations["user_id"].unique()
        user_ids = [int(u) for u in user_ids]
        cursor.execute(
            "DELETE FROM recommendations WHERE user_id = ANY(%s)", (list(user_ids),)
        )
        print(f"Deleted existing recommendations for {len(user_ids)} users.")
        con.commit()

        # insert new recommendations
        print("inserting new recommendations...")
        insert_query = """
            INSERT INTO recommendations (user_id, restaurant_id, score)
            VALUES %s
        """
        execute_values(cursor, insert_query, df_recommendations.values.tolist())
        print(f"Inserted {len(df_recommendations)} recommendations into DB")
        print("Sample inserted recommendations:")
        print(df_recommendations.head(5))
        con.commit()

        print("[6] write_DB process completed successfully.")

    except Exception as e:
        print(f"Error in write_DB: {e}")
        con.rollback()
        raise

# ------------------ 7. create view  -----------------------
def create_view(cursor,con):
    try:
        print("creating or replacing recommendation view...")
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
        print("[7] create_view process completed successfully.")
        cursor.close()
        print("Database connection closed.")
    except Exception as e:
        print(f"Error in create_view: {e}")
        raise