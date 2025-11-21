"""Streamlit demo for Item2vec recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap

from src.models.item2vec import Item2Vec
from src.models.baselines import PopularityRecommender, UserKNNRecommender, ItemKNNRecommender
from src.data.data_utils import generate_synthetic_data, create_item_sequences, encode_items, create_negative_samples
from src.utils.seed import set_seed, get_device


@st.cache_data
def load_data():
    """Load or generate data."""
    try:
        # Try to load existing data
        interactions_df = pd.read_csv("data/interactions.csv")
        items_df = pd.read_csv("data/items.csv")
        users_df = pd.read_csv("data/users.csv")
    except FileNotFoundError:
        # Generate synthetic data if not found
        st.info("No existing data found. Generating synthetic data...")
        interactions_df, items_df, users_df = generate_synthetic_data(
            n_users=500,
            n_items=200,
            n_interactions=5000,
            random_seed=42,
        )
    
    return interactions_df, items_df, users_df


@st.cache_resource
def load_models(interactions_df, items_df):
    """Load or train models."""
    try:
        # Try to load existing models
        item2vec_model = torch.load("models/item2vec_model.pth")
        item_encoder = torch.load("models/item_encoder.pth")
        
        # Load baseline models
        popularity_model = PopularityRecommender()
        popularity_model.fit(interactions_df)
        
        user_knn_model = UserKNNRecommender(k=50)
        user_knn_model.fit(interactions_df)
        
        item_knn_model = ItemKNNRecommender(k=50)
        item_knn_model.fit(interactions_df)
        
    except FileNotFoundError:
        # Train models if not found
        st.info("No existing models found. Training models...")
        
        # Set seed
        set_seed(42)
        
        # Train Item2vec
        train_sequences = create_item_sequences(interactions_df, window_size=5)
        all_items = list(set([item for seq in train_sequences for item in seq]))
        item_encoder, encoded_items = encode_items(all_items)
        
        train_positive_pairs = []
        for sequence in train_sequences:
            encoded_seq = [item_encoder.transform([item])[0] for item in sequence]
            for i in range(len(encoded_seq) - 1):
                train_positive_pairs.append((encoded_seq[i], encoded_seq[i + 1]))
        
        train_samples = create_negative_samples(
            train_positive_pairs,
            encoded_items,
            num_negative=5,
            random_seed=42,
        )
        
        device = get_device("auto")
        item2vec_model = Item2Vec(
            vocab_size=len(all_items),
            embedding_dim=32,
            device=device,
        )
        
        item2vec_model.train_model(
            train_samples=train_samples,
            num_epochs=50,
            batch_size=512,
            verbose=False,
        )
        
        # Train baselines
        popularity_model = PopularityRecommender()
        popularity_model.fit(interactions_df)
        
        user_knn_model = UserKNNRecommender(k=50)
        user_knn_model.fit(interactions_df)
        
        item_knn_model = ItemKNNRecommender(k=50)
        item_knn_model.fit(interactions_df)
    
    return item2vec_model, item_encoder, popularity_model, user_knn_model, item_knn_model


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Item2vec Recommendation System",
        page_icon="🎯",
        layout="wide",
    )
    
    st.title("🎯 Item2vec Recommendation System Demo")
    st.markdown("Explore item similarities and get personalized recommendations using Item2vec embeddings.")
    
    # Load data
    interactions_df, items_df, users_df = load_data()
    
    # Load models
    item2vec_model, item_encoder, popularity_model, user_knn_model, item_knn_model = load_models(interactions_df, items_df)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Item Similarity Explorer", "User Recommendations", "Model Comparison", "Data Overview"]
    )
    
    if page == "Item Similarity Explorer":
        st.header("🔍 Item Similarity Explorer")
        
        # Item selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_item = st.selectbox(
                "Select an item to explore:",
                items_df["item_id"].tolist(),
                format_func=lambda x: f"{x} - {items_df[items_df['item_id'] == x]['title'].iloc[0]}"
            )
        
        with col2:
            top_k = st.slider("Number of similar items:", 5, 20, 10)
        
        if selected_item:
            # Get similar items using Item2vec
            try:
                item_idx = item_encoder.transform([selected_item])[0]
                similar_items = item2vec_model.get_similar_items(item_idx, top_k=top_k)
                
                st.subheader(f"Items similar to {selected_item}")
                
                # Display similar items
                for i, (similar_item_idx, similarity) in enumerate(similar_items):
                    similar_item_id = item_encoder.inverse_transform([similar_item_idx])[0]
                    similar_item_info = items_df[items_df["item_id"] == similar_item_id].iloc[0]
                    
                    with st.expander(f"{i+1}. {similar_item_id} - {similar_item_info['title']} (Similarity: {similarity:.3f})"):
                        st.write(f"**Category:** {similar_item_info['category']}")
                        st.write(f"**Price:** ${similar_item_info['price']}")
                        st.write(f"**Description:** {similar_item_info['description']}")
                
                # Visualization
                st.subheader("Similarity Visualization")
                
                # Get embeddings for visualization
                embeddings = item2vec_model.get_item_embeddings().cpu().numpy()
                
                # Reduce dimensionality
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                
                # Create visualization
                fig = go.Figure()
                
                # Plot all items
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[:, 0],
                    y=embeddings_2d[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='lightblue', opacity=0.6),
                    text=[f"{item_encoder.inverse_transform([i])[0]}" for i in range(len(embeddings))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    name="All Items"
                ))
                
                # Highlight selected item
                selected_embedding = embeddings_2d[item_idx]
                fig.add_trace(go.Scatter(
                    x=[selected_embedding[0]],
                    y=[selected_embedding[1]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    text=[selected_item],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    name="Selected Item"
                ))
                
                # Highlight similar items
                similar_indices = [idx for idx, _ in similar_items]
                similar_embeddings = embeddings_2d[similar_indices]
                fig.add_trace(go.Scatter(
                    x=similar_embeddings[:, 0],
                    y=similar_embeddings[:, 1],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond'),
                    text=[item_encoder.inverse_transform([idx])[0] for idx in similar_indices],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    name="Similar Items"
                ))
                
                fig.update_layout(
                    title="Item Embeddings Visualization (PCA)",
                    xaxis_title="First Principal Component",
                    yaxis_title="Second Principal Component",
                    showlegend=True,
                    height=500,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error getting similar items: {e}")
    
    elif page == "User Recommendations":
        st.header("👤 User Recommendations")
        
        # User selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_user = st.selectbox(
                "Select a user:",
                users_df["user_id"].tolist()
            )
        
        with col2:
            top_k = st.slider("Number of recommendations:", 5, 20, 10)
        
        if selected_user:
            # Get user's interaction history
            user_interactions = interactions_df[interactions_df["user_id"] == selected_user]
            
            st.subheader(f"Recommendations for {selected_user}")
            
            # Show user's interaction history
            if not user_interactions.empty:
                st.write("**User's Interaction History:**")
                user_items = user_interactions.merge(items_df, on="item_id")
                st.dataframe(user_items[["item_id", "title", "category", "rating"]], use_container_width=True)
            else:
                st.write("No interaction history found for this user.")
            
            # Get recommendations from different models
            models = {
                "Item2vec": item2vec_model,
                "Popularity": popularity_model,
                "User-kNN": user_knn_model,
                "Item-kNN": item_knn_model,
            }
            
            for model_name, model in models.items():
                st.subheader(f"{model_name} Recommendations")
                
                try:
                    if model_name == "Item2vec":
                        # For Item2vec, we need to implement a recommendation method
                        # For now, we'll use item similarity based on user's interacted items
                        if not user_interactions.empty:
                            user_items = user_interactions["item_id"].tolist()
                            recommendations = []
                            
                            for item_id in user_items[:5]:  # Use first 5 items
                                try:
                                    item_idx = item_encoder.transform([item_id])[0]
                                    similar_items = item2vec_model.get_similar_items(item_idx, top_k=top_k)
                                    recommendations.extend([item_encoder.inverse_transform([idx])[0] for idx, _ in similar_items])
                                
                                except ValueError:
                                    continue
                            
                            # Remove duplicates and user's own items
                            recommendations = list(set(recommendations))
                            recommendations = [r for r in recommendations if r not in user_items]
                            recommendations = recommendations[:top_k]
                        else:
                            recommendations = []
                    else:
                        recommendations = model.recommend(selected_user, top_k=top_k)
                    
                    if recommendations:
                        rec_df = pd.DataFrame(recommendations, columns=["item_id"])
                        rec_df = rec_df.merge(items_df, on="item_id")
                        st.dataframe(rec_df[["item_id", "title", "category", "price"]], use_container_width=True)
                    else:
                        st.write("No recommendations available.")
                        
                except Exception as e:
                    st.error(f"Error getting recommendations from {model_name}: {e}")
    
    elif page == "Model Comparison":
        st.header("📊 Model Comparison")
        
        # This would typically show evaluation metrics
        st.write("Model comparison metrics would be displayed here.")
        st.write("Run the training script to generate evaluation results.")
        
        # Show sample evaluation results
        st.subheader("Sample Evaluation Results")
        
        # Create sample data for demonstration
        sample_results = {
            "Model": ["Item2vec", "Popularity", "User-kNN", "Item-kNN"],
            "Precision@10": [0.125, 0.089, 0.142, 0.156],
            "Recall@10": [0.234, 0.167, 0.267, 0.289],
            "NDCG@10": [0.187, 0.134, 0.201, 0.218],
            "Hit Rate@10": [0.456, 0.323, 0.489, 0.512],
        }
        
        results_df = pd.DataFrame(sample_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Precision@10", "Recall@10", "NDCG@10", "Hit Rate@10"],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ["Precision@10", "Recall@10", "NDCG@10", "Hit Rate@10"]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=results_df["Model"],
                    y=results_df[metric],
                    name=metric,
                    showlegend=False,
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Overview":
        st.header("📈 Data Overview")
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        
        with col2:
            st.metric("Total Items", len(items_df))
        
        with col3:
            st.metric("Total Interactions", len(interactions_df))
        
        with col4:
            st.metric("Avg Rating", f"{interactions_df['rating'].mean():.2f}")
        
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = interactions_df["rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Distribution of Ratings",
            labels={"x": "Rating", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        st.subheader("Item Category Distribution")
        category_counts = items_df["category"].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Item Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # User activity
        st.subheader("User Activity Distribution")
        user_activity = interactions_df.groupby("user_id").size()
        fig = px.histogram(
            x=user_activity.values,
            title="Distribution of User Activity (Interactions per User)",
            labels={"x": "Number of Interactions", "y": "Number of Users"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        
        tab1, tab2, tab3 = st.tabs(["Interactions", "Items", "Users"])
        
        with tab1:
            st.dataframe(interactions_df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(items_df.head(10), use_container_width=True)
        
        with tab3:
            st.dataframe(users_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
