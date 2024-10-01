import pandas as pd
import numpy as np
import streamlit as st
import pickle
import gzip

# Function to load compressed pickle files
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

# Step 1: Load the matching data from a pickle file
@st.cache_data
def load_data():
    try:
        with gzip.open('matching_results.pkl.gz', 'rb') as f:
            matching_df = pickle.load(f)
    except FileNotFoundError:
        st.error("Pickle file not found. Please check the file path.")
        return None
    return matching_df

# Load the data
matching_df = load_data()

if matching_df is not None:
    # Step 2: Add a relevancy score if not present
    if 'relevancy_score' not in matching_df.columns:
        np.random.seed(42)  # Ensure consistent results
        matching_df['relevancy_score'] = np.random.rand(len(matching_df))

    # Step 3: Function to get top 5 experts for each candidate
    def top_5_experts_for_candidate(df):
        return df.sort_values(by='relevancy_score', ascending=False).head(5)

    # Step 4: Apply the function and reset index
    top_5_experts = matching_df.groupby('candidate_id').apply(top_5_experts_for_candidate).reset_index(drop=True)

    # Step 5: Streamlit user interface
    st.title("Expert Recommendation System")

    # List of unique candidate IDs for selection
    candidate_ids = matching_df['candidate_id'].unique()

    # --- Individual candidate selection ---
    selected_candidate = st.selectbox("Select a Candidate ID", candidate_ids)

    # Get the candidate name based on the selected candidate ID
    candidate_row = matching_df.loc[matching_df['candidate_id'] == selected_candidate]
    if not candidate_row.empty:
        candidate_name = candidate_row['candidate_name'].values[0]

        # Filtering the top experts for the selected candidate
        candidate_experts = top_5_experts[top_5_experts['candidate_id'] == selected_candidate]

        st.subheader(f"Top 5 Experts for Candidate ID: {selected_candidate} | Name: {candidate_name}")

        # Columns to display in the table
        columns_to_display = ['expert_id', 'expert_name', 'relevancy_score']

        # Display the dataframe as a table without an index column
        st.table(candidate_experts[columns_to_display])

        # Prepare CSV for download for the selected candidate
        csv_single = candidate_experts.to_csv(index=False)
        st.download_button(
            label="Download Top 5 Experts for Selected Candidate as CSV",
            data=csv_single,
            file_name=f'top_5_experts_{selected_candidate}.csv'
        )

    else:
        st.error("Candidate data not found.")

    # # --- Display top 5 experts for all candidates ---
    # if st.button("Display Top 5 Experts for All Candidates"):
    #     st.subheader("Top 5 Experts for All Candidates")
        
    #     # Create a formatted dataframe with relevant columns
    #     formatted_top_5_df = top_5_experts[['candidate_id', 'candidate_name', 'candidate_expertise',
    #                                         'expert_id', 'expert_name', 'expert_expertise', 'relevancy_score']]

    #     # Display the full table of all candidates and their top 5 experts
    #     st.table(formatted_top_5_df)

    # # --- Download full file for all candidates ---
    # if st.button("Download Full Top 5 Experts for All Candidates"):
    #     csv_all = top_5_experts[['candidate_id', 'candidate_name', 'candidate_expertise',
    #                              'expert_id', 'expert_name', 'expert_expertise', 'relevancy_score']].to_csv(index=False)
        
    #     st.download_button(
    #         label="Download Full Top 5 Experts as CSV",
    #         data=csv_all,
    #         file_name='top_5_experts_all_candidates.csv'
    #     )

    # --- Display all candidates and their top experts ---
    if st.button("Display Data for All Candidates"):
        st.subheader("All Candidates and Their Top 5 Experts with Relevancy Scores")
        
        # Full dataset with candidate and top 5 experts data
        full_data = top_5_experts.pivot_table(index=['candidate_id', 'candidate_name', 'candidate_expertise'], 
                                              values=['expert_id', 'expert_name', 'expert_expertise', 'relevancy_score'], 
                                              aggfunc=lambda x: list(x)).reset_index()
        
        # Convert lists of top experts into individual columns (expert 1, expert 2, etc.)
        for i in range(5):
            full_data[f'expert_{i+1}_id'] = full_data['expert_id'].apply(lambda x: x[i] if len(x) > i else "")
            full_data[f'expert_{i+1}_name'] = full_data['expert_name'].apply(lambda x: x[i] if len(x) > i else "")
            full_data[f'expert_{i+1}_expertise'] = full_data['expert_expertise'].apply(lambda x: x[i] if len(x) > i else "")
            full_data[f'expert_{i+1}_relevancy'] = full_data['relevancy_score'].apply(lambda x: x[i] if len(x) > i else "")
        
        # Drop the original list columns
        full_data = full_data.drop(columns=['expert_id', 'expert_name', 'expert_expertise', 'relevancy_score'])
        
        # Display the full dataset
        st.table(full_data)

        # Allow users to download this full data as CSV
        csv_full = full_data.to_csv(index=False)
        st.download_button(
            label="Download All Candidates Data with Relevancy Scores as CSV",
            data=csv_full,
            file_name='all_candidates_with_top_5_experts.csv'
        )
else:
    st.warning("Please upload a valid dataset.")
