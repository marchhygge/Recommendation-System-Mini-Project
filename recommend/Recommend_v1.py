import pandas as pd
import numpy as np
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from os.path import join, dirname
import os

try:
    # Load environment variables from .env file
    dotenv_path = join(dirname(__file__), '.env')
    if not dotenv_path:
        raise FileNotFoundError(".env file not found.")
    load_dotenv(dotenv_path)
    if load_dotenv:
        print("[1] Environment variables loaded successfully.")

    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    USER = os.getenv("USER")
    PASSWORD = os.getenv("PASSWORD")
    
    # Connect to PostgreSQL database
    connection = psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=USER,
        password=PASSWORD
    )
    cursor = connection.cursor()
    if not connection:
        raise ConnectionError("Failed to connect to PostgreSQL database.")
    else:
        print("[2] Connected to PostgreSQL database successfully.")

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
        inner join restaurant_tags b
            on a.id = b.restaurant_id
        inner join tags c
            on c.id = b.tag_id
        inner join categories d 
            on d.id = c.category_id
    """
    # Get restaurants
    cursor.execute(restaurant_query)
    if cursor is None:
        raise ValueError("Cursor is None after executing restaurant query.")
    else:
        print("[3] Restaurant query executed successfully.")
    records_restaurant = cursor.fetchall()
    if len(records_restaurant) == 0:
        raise ValueError("No restaurant data fetched.")
    else:
        df_restaurants = pd.DataFrame(records_restaurant
                                     ,columns=[desc[0] for desc in cursor.description])
        print("sample restaurant data:")
        print(df_restaurants.head(5))
        
    # Query users
    users_query = """
        select b.user_id,
               a.username,
               c.category_id,
               d.name as category_name,
               b.tag_id,
               c.name as tag_name
        from users a
        inner join user_tags b
            on b.user_id = a.id
        inner join tags c
            on c.id = b.tag_id
        inner join categories d 
            on d.id = c.category_id
    """
    # Get users
    cursor.execute(users_query)
    if cursor is None:
        raise ValueError("Cursor is None after executing users query.")
    else:
        print("[4] users query executed successfully.")
    records_users = cursor.fetchall()
    if len(records_users) == 0:
        raise ValueError("No user data fetched.")
    else:
        df_users = pd.DataFrame(records_users
                                ,columns=[desc[0] for desc in cursor.description])
        print("sample users data:")
        print(df_users.head(5))

    # Create profiles
    print("[5] Creating profiles...")
    if df_restaurants.empty or df_users.empty:
        raise ValueError("No data fetched from the database.")
    else:
        restaurant_metrics = df_restaurants.groupby(["restaurant_id"])["tag_name"] \
                                           .apply(lambda x: ' | '.join(x)) \
                                           .reset_index(name="restaurantCharacteristics")
        print("sample restaurant profile data:")
        print(restaurant_metrics.head(5))
        
        user_metrics = df_users.groupby(["user_id"])["tag_name"] \
                               .apply(lambda x: " ".join(x)) \
                               .reset_index(name="userCharacteristics")
        print("sample user profile data:")
        print(user_metrics.head(5))

    # TF-IDF Vectorization
    print("[6] Performing TF-IDF vectorization...")
    if restaurant_metrics.empty or user_metrics.empty:
        raise ValueError("No data available for TF-IDF vectorization.")
    else:
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Fit all data to ensure consistent feature space
        all_text = pd.concat([user_metrics["userCharacteristics"]
                            , restaurant_metrics["restaurantCharacteristics"]])
        vectorizer.fit(all_text)

        # Transform metrics
        user_vecs = vectorizer.transform(user_metrics["userCharacteristics"])
        restaurant_vecs = vectorizer.transform(restaurant_metrics["restaurantCharacteristics"])

        # Compute cosine similarity
        print("[7] Computing cosine similarity...")
        if user_vecs.shape[0] == 0 or restaurant_vecs.shape[0] == 0:
            raise ValueError("TF-IDF vectorization resulted in empty matrices.")
        else:
            similarity = cosine_similarity(user_vecs, restaurant_vecs)
        
        # get top-k recommendations
        print("[8] Generating recommendations...")
        k = 5  # number of recommendations per user
        recommendations = []
        for user_idx, user_id in enumerate(user_metrics["user_id"]):
            # get top-k indices
            top_k_idx = np.argpartition(-similarity[user_idx], k)[:k]
            # sort top-k indices by similarity score
            top_k_sorted = top_k_idx[np.argsort(-similarity[user_idx][top_k_idx])]
            for restaurant_idx in top_k_sorted:
                recommendations.append((
                    int(user_id),
                    int(restaurant_metrics.iloc[restaurant_idx]["restaurant_id"]),
                    float(similarity[user_idx, restaurant_idx])
                ))

        df_recommendations = pd.DataFrame(recommendations
                                        , columns=["user_id", "restaurant_id", "score"])
        print("sample recommendations data:")
        print(df_recommendations.head(5))
        
        if df_recommendations.empty:
            raise ValueError("No recommendations to insert into the database.")

        # Clear existing recommendations for the users
        print("[9] Deleting existing recommendations...")         
        user_ids = df_recommendations["user_id"].unique()
        user_ids = [int(u) for u in user_ids]
        cursor.execute("DELETE FROM recommendation WHERE user_id = ANY(%s)", (list(user_ids),))
        print(f"Deleted existing recommendations for {len(user_ids)} users.")
        connection.commit()

        # Insert new recommendations
        print("[10] Inserting new recommendations...")
        insert_query = """
            INSERT INTO recommendation (user_id, restaurant_id, score)
            VALUES %s
        """
        execute_values(cursor
                      ,insert_query
                      ,df_recommendations.values.tolist())
        print(f"Inserted {len(df_recommendations)} recommendations into DB")
        print("Sample inserted recommendations:")
        print(df_recommendations.head(5))

        # create of replace recommendation view
        print("[11] Creating or replacing recommendation view...")
        cursor.execute("""
            CREATE OR REPLACE VIEW recommendation_view AS
            SELECT r.id as recommendation_id,
                u.username as user_name,
                res.name as restaurant_name,
                r.score
            FROM recommendation r
            JOIN users u ON r.user_id = u.id
            JOIN restaurants res ON r.restaurant_id = res.id
        """)
        print("Recommendation view created or replaced successfully.")

        connection.commit()
        print("All changes committed to the database.")
        print("Recommendation process completed successfully.")

except Exception as e:
    print(f"Error: {e}")
    if connection:
        connection.rollback()
        print("Transaction rolled back due to error.")
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")