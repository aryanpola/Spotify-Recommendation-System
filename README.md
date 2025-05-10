# Spotify Recommendation System

This project implements a Spotify song recommendation system that uses user listening history and preferences to suggest new music. It leverages Spotify API data, FAISS for similarity search, and an epsilon-greedy bandit algorithm for personalizing recommendations based on user feedback.

## Project Structure


- **[final_notebook.ipynb](final_notebook.ipynb)**: The main Jupyter notebook to run the recommendation system.
- **[LICENSE](LICENSE)**: The MIT license for this project.
- **[requirements.txt](requirements.txt)**: A list of Python dependencies required to run the project.
- **[source_weights.json](source_weights.json)**: Stores the weights for different recommendation sources (e.g., top tracks, recent tracks). These weights are updated based on user feedback.
- **[Spotify_Recommender_Main.py](Spotify_Recommender_Main.py)**: A Python module containing the core logic for Spotify authentication, data fetching, data processing, recommendation generation, and feedback collection.

## Setup

1.  **Clone the repository (if applicable) or download the files.**
2.  **Install dependencies:**
    Open your terminal and navigate to the project directory. Run the following command to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Spotify API Credentials:**
    *   You need to create a Spotify Developer application to get a `Client ID` and `Client Secret`.
    *   Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/) and create an app.
    *   Once created, note down your `Client ID` and `Client Secret`.
    *   In your app settings on the Spotify Developer Dashboard, set the **Redirect URI** to `http://127.0.0.1:8888/callback`.
    *   Create a file named `spotify_config.json` in the root of the project directory with the following content, replacing `YOUR_CLIENT_ID` and `YOUR_CLIENT_SECRET` with your actual credentials:
        ```json
        {
            "client_id": "YOUR_CLIENT_ID",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uri": "http://127.0.0.1:8888/callback",
            "scope": "user-read-recently-played user-top-read user-library-read"
        }
        ```
4.  **Dataset:**
    *   The project uses a dataset of Spotify songs with attributes and lyrics. The notebook expects a file named `songs_with_attributes_and_lyrics.csv`.
    *   You can download a suitable dataset from Kaggle, for example: [960K Spotify Songs With Lyrics data](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics?select=songs_with_lyrics_and_timestamps.csv).
    *   Place the `songs_with_attributes_and_lyrics.csv` file in the root of the project directory.

## How to Run

1.  **Open the Jupyter Notebook:**
    Launch Jupyter Lab or Jupyter Notebook and open the [final_notebook.ipynb](final_notebook.ipynb) file.
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
2.  **Run the Cells:**
    Execute the cells in the notebook sequentially.
    *   The initial cells will import necessary libraries and process the song dataset.
    *   You will then be prompted to authenticate with Spotify. A web browser window will open asking you to log in and authorize the application.
    *   After successful authentication, the notebook will fetch your Spotify data (top tracks, recent tracks, etc.).
    *   Recommendations will be generated based on your data and the initial `source_weights.json`.
    *   You will be prompted in the notebook's output to rate the recommended songs.
    *   Based on your feedback, the `source_weights.json` file will be updated.

## Key Components in `Spotify_Recommender_Main.py`

*   **[`SpotifyAuthServer`](Spotify_Recommender_Main.py)**: Handles the OAuth2 authentication flow with Spotify.
*   **[`SpotifyUserData`](Spotify_Recommender_Main.py)**: Interacts with the Spotify API to fetch user-specific data like top tracks, recently played songs, saved tracks, and artist information.
*   **[`SpotifyUserDataFetcher`](Spotify_Recommender_Main.py)**: Uses `SpotifyUserData` to fetch and process various types of user data, and ranks tracks based on source weights and time decay.
*   **[`HelperFunctions`](Spotify_Recommender_Main.py)**: Contains utility functions for tasks like token validation, token refreshing, data conversion, and calculating exponential decay for time-sensitive data.
*   **[`SpotifyDataEnricher`](Spotify_Recommender_Main.py)**: Enriches song data in a DataFrame by fetching additional details like genres, popularity, and explicit content from the Spotify API.
*   **[`EpsilonGreedyBandit`](Spotify_Recommender_Main.py)**: Implements an epsilon-greedy multi-armed bandit algorithm to dynamically update the weights of different recommendation sources based on user feedback.
*   **[`RecommendationFeedbackCollector`](Spotify_Recommender_Main.py)**: Manages the process of presenting recommendations to the user and collecting their ratings.

## How it Works

1.  **Data Preparation**: A large dataset of songs with their audio features is loaded and preprocessed. A FAISS index is built on these features for efficient similarity search.
2.  **Spotify Authentication**: The user authenticates with their Spotify account through an OAuth2 flow managed by a local Flask server.
3.  **User Data Fetching**: The system fetches the user's top tracks, recently played songs, saved tracks, and top artists from the Spotify API.
4.  **User Profile Generation**:
    *   Tracks from different sources (top, recent, saved, top artists' songs) are scored based on predefined weights in `source_weights.json` and time decay for recent/saved tracks.
    *   A weighted average feature vector (user profile) is created for each source based on the features of the highly-ranked songs from that source.
5.  **Recommendation Generation**:
    *   For each source-specific user profile vector, the FAISS index is queried to find the most similar songs from the main dataset. These are the initial recommendations for that source.
6.  **Feedback Collection**:
    *   The system presents recommendations to the user (e.g., 10 songs per source).
    *   The user rates these songs (e.g., on a scale of 0-10).
7.  **Weight Update (Reinforcement Learning)**:
    *   The [`EpsilonGreedyBandit`](Spotify_Recommender_Main.py) algorithm uses the user's feedback to update the `source_weights.json`. Sources that provide songs the user rates highly will have their weights increased, while sources providing less liked songs will have their weights decreased.
    *   This allows the system to learn which sources are more relevant for the user over time.
8.  **Saving Updated Weights**: The new source weights are saved back to `source_weights.json` for future sessions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
