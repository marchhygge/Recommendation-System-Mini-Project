import pandas as pd
import numpy as np
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # Connect to PostgreSQL database
    connection = psycopg2.connect(
        host="db.pqebyefuvarvwvwovchh.supabase.co",
        database="postgres",
        user="postgres",
        password="Hoangthinh@2004"
    )
    cursor = connection.cursor()
    if not connection:
        raise ConnectionError("Failed to connect to PostgreSQL database.")

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
    records_restaurant = cursor.fetchall()
    if len(records_restaurant) == 0:
        raise ValueError("No restaurant data fetched.")
    else:
        df_restaurants = pd.DataFrame(records_restaurant
                                        ,columns=[desc[0] for desc in cursor.description])
        
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
    records_users = cursor.fetchall()
    if len(records_users) == 0:
        raise ValueError("No user data fetched.")
    else:
        df_users = pd.DataFrame(records_users
                                ,columns=[desc[0] for desc in cursor.description])

    # Create profiles
    if df_restaurants.empty or df_users.empty:
        raise ValueError("No data fetched from the database.")
    else:
        restaurant_metrics = df_restaurants.groupby(["restaurant_id"])["tag_name"] \
                                           .apply(lambda x: ' | '.join(x)) \
                                           .reset_index(name="restaurantCharacteristics")
        if restaurant_metrics.empty:
            raise ValueError("No restaurant profile data available after grouping.")
        
        user_metrics = df_users.groupby(["user_id"])["tag_name"] \
                               .apply(lambda x: ' | '.join(x)) \
                               .reset_index(name="userCharacteristics")
        if user_metrics.empty:
            raise ValueError("No user profile data available after grouping.")

    # TF-IDF Vectorization
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

        # Compute cosine similarity]
        if user_vecs.shape[0] == 0 or restaurant_vecs.shape[0] == 0:
            raise ValueError("TF-IDF vectorization resulted in empty matrices.")
        else:
            similarity = cosine_similarity(user_vecs, restaurant_vecs)
        
        # get top-k recommendations
        k = 5  # number of recommendations per user
        recommendations = []
        for user_idx, user_id in enumerate(user_metrics["user_id"]):
            # get top-k indices
            top_k_idx = np.argpartition(-similarity[user_idx], k)[:k]
            # sort top-k indices by similarity score
            top_k_sorted = top_k_idx[np.argsort(-similarity[user_idx][top_k_idx])]
            for restaurant_idx in top_k_sorted:
                recommendations.append((
                    str(user_id),
                    str(restaurant_metrics.iloc[restaurant_idx]["restaurant_id"]),
                    float(similarity[user_idx, restaurant_idx])
                ))

        df_recommendations = pd.DataFrame(recommendations
                                        , columns=["user_id", "restaurant_id", "score"])
        
        if df_recommendations.empty:
            raise ValueError("No recommendations to insert into the database.")
        
        # Insert recommendations into the database
        insert_query = """
            INSERT INTO recommendation (user_id, restaurant_id, score)
            VALUES (%s, %s, %s)
        """

        # Clear existing recommendations for the users
        user_ids = df_recommendations["user_id"].unique()
        cursor.execute("DELETE FROM recommendation WHERE user_id = ANY(%s)", (list(user_ids),))
        connection.commit()

        # Insert new recommendations
        cursor.executemany(insert_query, df_recommendations.values.tolist())
        connection.commit()
        
        print(f"Inserted {len(df_recommendations)} recommendations into DB")

except Exception as e:
    print(f"Error: {e}")
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")