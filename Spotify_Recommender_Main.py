# Standard and third-party library imports (import ...)
import base64
import json
import math
import random
import threading
import time
import urllib.parse
import webbrowser
import faiss
import numpy as np
import pandas as pd
import requests

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from flask import Flask, request
from typing import Any, Dict, List, Optional, Tuple, Union



class SpotifyAuthServer:
    """
    A server to handle Spotify OAuth authentication, token exchange, and Flask app setup.
    This class is responsible for:
    1. Handling Spotify login flow via OAuth.
    2. Receiving the authorization code from the callback URL.
    3. Exchanging the authorization code for an access token.
    4. Running a Flask web server to interact with the user.

    Attributes:
        config_path (str): Path to the Spotify configuration file.
        auth_code (str | None): The authorization code received after login.
        token_data (dict | None): The token information after successful token exchange.
        _config (dict): Configuration loaded from the provided config file.
        _client_id (str): Spotify client ID.
        _client_secret (str): Spotify client secret.
        _redirect_uri (str): Spotify redirect URI.
        _scope (str): Spotify authorization scope.
        _app (Flask): Flask web server instance.
    """
    def __init__(self, config_path: str = 'spotify_config.json') -> None:
        """
        Initializes the SpotifyAuthServer class, loads the config, and sets up the Flask app.

        Args:
            config_path (str): Path to the Spotify config file.
        """
        self.config_path = config_path
        self.auth_code = None  # Public: to be accessed from outside
        self.token_data = None  # Public: will hold token info after exchange

        # Private config loading
        self._config = self._load_config(self.config_path)
        self._client_id = self._config['client_id']
        self._client_secret = self._config['client_secret']
        self._redirect_uri = self._config['redirect_uri']
        self._scope = self._config['scope']

        # Flask app and routes
        self._app = Flask(__name__)
        self._register_routes()

    def _load_config(self, path: str) -> dict:
        """
        Loads configuration data from a specified file.

        Args:
            path (str): Path to the config file.
        
        Returns:
            dict: Loaded configuration.
        """
        with open(path, 'r') as f:
            return json.load(f)

    def _register_routes(self) -> None:
        """
        Registers routes for the Flask app to handle login and callback.
        """
        self._app.add_url_rule('/', view_func=self.login)
        self._app.add_url_rule('/callback', view_func=self._callback)

    def _build_auth_url(self) -> str:
        """
        Constructs the URL for Spotify's authorization endpoint.

        Returns:
            str: The Spotify authorization URL with query parameters.
        """
        query_params = {
            'client_id': self._client_id,
            'response_type': 'code',
            'redirect_uri': self._redirect_uri,
            'scope': self._scope
        }
        return 'https://accounts.spotify.com/authorize?' + urllib.parse.urlencode(query_params)

    def login(self) -> str:
        """
        Initiates the Spotify login process by opening the authorization URL.

        Returns:
            str: Message indicating redirection to Spotify login.
        """
        auth_url = self._build_auth_url()
        webbrowser.open(auth_url)
        return 'Redirecting to Spotify login...'

    def _callback(self) -> str:
        """
        Handles the callback from Spotify after the user has logged in.
        Extracts the authorization code from the URL parameters.

        Returns:
            str: Message indicating the received authorization code.
        """
        code = request.args.get('code')
        self.auth_code = code
        return f'Authorization code received: {code}'

    def exchange_token(self) -> dict:
        """
        Exchanges the authorization code for an access token from Spotify.

        Returns:
            dict: The token data containing access token and related info.

        Raises:
            HTTPError: If the token exchange fails.
        """
        token_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()

        headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {
            'grant_type': 'authorization_code',
            'code': self.auth_code,
            'redirect_uri': self._redirect_uri
        }

        response = requests.post(token_url, headers=headers, data=data)

        if response.status_code == 200:
            self.token_data = response.json()
            return self.token_data
        else:
            print("Error exchanging token:", response.text)
            response.raise_for_status()

    def run(self, host: str = '127.0.0.1', port: int = 8888) -> None:
        """
        Runs the Flask app, starting the Spotify authentication server.

        Args:
            host (str): The host to run the Flask server on.
            port (int): The port to run the Flask server on.
        """
        print(f"Running Spotify Auth Server on http://{host}:{port}/")
        self._app.run(host=host, port=port)


class SpotifyUserData:
    """
    A class to interact with the Spotify API and retrieve various user-specific data, including:
    1. User profile (e.g., display name, email).
    2. Recently played tracks.
    3. User's top tracks and artists over specified time ranges.
    4. Saved tracks.
    5. Audio features of tracks (e.g., danceability, energy).
    6. Track metadata (e.g., album, artists, duration).
    7. Artist genres.
    8. Song details including genres, popularity, and explicit content.
    9. Top songs by artist.

    Attributes:
        access_token (str): Access token for Spotify API authentication.
        refresh_token (str): Refresh token for renewing access token.
        client_id (str): Spotify client ID.
        client_secret (str): Spotify client secret.
    """
    
    def __init__(self, access_token: str, refresh_token: str, client_id: str, client_secret: str) -> None:
        """
        Initializes the SpotifyUserData class with access tokens, client ID, and client secret.

        Args:
            access_token (str): The access token to authenticate with the Spotify API.
            refresh_token (str): The refresh token for token renewal.
            client_id (str): Spotify client ID.
            client_secret (str): Spotify client secret.
        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._headers = {'Authorization': f'Bearer {self._access_token}'}

    @property
    def access_token(self) -> str:
        """Getter for the access token."""
        return self._access_token

    @access_token.setter
    def access_token(self, token: str) -> None:
        """Setter for the access token."""
        self._access_token = token

    def _get_headers(self) -> dict:
        """Generates the headers used for Spotify API requests."""
        return {'Authorization': f'Bearer {self._access_token}'}

    def _get(self, url: str) -> dict:
        """
        Makes a GET request to the given URL and handles retries or token refresh.

        Args:
            url (str): The Spotify API URL to send the GET request to.
        
        Returns:
            dict: The JSON response from the Spotify API.
        
        Raises:
            HTTPError: If an error occurs during the request.
        """
        while True:
            response = requests.get(url, headers=self._get_headers())
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after)
                continue
            elif response.status_code == 401:
                self._refresh_access_token()  # Refresh the token if it's expired
                continue  # Retry the request with the new token
            elif response.status_code != 200:
                raise requests.exceptions.HTTPError(f"Spotify API error {response.status_code}: {response.text}")
            
            return response.json()

    def _refresh_access_token(self) -> None:
        """
        Refreshes the access token using the refresh token and updates the authorization headers.

        Raises:
            Exception: If the refresh token request fails.
        """
        url = 'https://accounts.spotify.com/api/token'
        
        # Prepare the data for refreshing the token
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self._refresh_token,
            'client_id': self._client_id,
            'client_secret': self._client_secret
        }

        response = requests.post(url, data=data)

        if response.status_code == 200:
            # Extract the new access token from the response
            new_token_data = response.json()
            self._access_token = new_token_data['access_token']
            self._headers = {'Authorization': f'Bearer {self._access_token}'}  # Update headers with new token
        else:
            raise Exception(f"Failed to refresh access token: {response.status_code} - {response.text}")

    def get_user_profile(self) -> dict:
        """
        Retrieves the current user's profile information (e.g., id, display_name, email).

        Returns:
            dict: User profile data.
        """
        return self._get('https://api.spotify.com/v1/me')

    def get_recently_played(self, limit: int = 20) -> dict:
        """
        Retrieves the current user's recently played tracks.

        Args:
            limit (int): The number of recently played tracks to retrieve (default is 20).

        Returns:
            dict: Recently played track data.
        """
        url = f'https://api.spotify.com/v1/me/player/recently-played?limit={limit}'
        return self._get(url)

    def get_top_tracks(self, time_range: str = 'medium_term', limit: int = 20) -> dict:
        """
        Retrieves the current user's top tracks over a specified time range.

        Args:
            time_range (str): Time range for top tracks (default is 'medium_term').
            limit (int): The number of top tracks to retrieve (default is 20).

        Returns:
            dict: Top tracks data.
        """
        url = f'https://api.spotify.com/v1/me/top/tracks?limit={limit}&time_range={time_range}'
        return self._get(url)

    def get_saved_tracks(self, limit: int = 20, offset: int = 0) -> dict:
        """
        Retrieves the current user's saved tracks.

        Args:
            limit (int): The number of saved tracks to retrieve (default is 20).
            offset (int): The offset for pagination (default is 0).

        Returns:
            dict: Saved tracks data.
        """
        url = f'https://api.spotify.com/v1/me/tracks?limit={limit}&offset={offset}'
        return self._get(url)

    def get_top_artists(self, time_range: str = 'medium_term', limit: int = 20) -> dict:
        """
        Retrieves the current user's top artists over a specified time range.

        Args:
            time_range (str): Time range for top artists (default is 'medium_term').
            limit (int): The number of top artists to retrieve (default is 20).

        Returns:
            dict: Top artists data.
        """
        url = f'https://api.spotify.com/v1/me/top/artists?limit={limit}&time_range={time_range}'
        return self._get(url)

    def get_audio_features(self, track_ids: Union[str, List[str]]) -> dict:
        """
        Retrieves audio features for the given track IDs.

        Args:
            track_ids (Union[str, List[str]]): A single track ID or a list of track IDs.
        
        Returns:
            dict: Audio features data.
        
        Raises:
            ValueError: If more than 100 track IDs are provided.
        """
        if isinstance(track_ids, list):
            if len(track_ids) > 100:
                raise ValueError("A maximum of 100 track IDs is allowed per request.")
            track_ids = ','.join(track_ids)
        url = f'https://api.spotify.com/v1/audio-features?ids={track_ids}'
        return self._get(url)

    def get_track_metadata(self, track_ids: Union[str, List[str]]) -> dict:
        """
        Retrieves metadata for the given track IDs.

        Args:
            track_ids (Union[str, List[str]]): A single track ID or a list of track IDs.
        
        Returns:
            dict: Track metadata data.
        
        Raises:
            ValueError: If more than 50 track IDs are provided.
        """
        if isinstance(track_ids, list):
            if len(track_ids) > 50:
                raise ValueError("A maximum of 50 track IDs is allowed per request.")
            track_ids = ','.join(track_ids)
        url = f'https://api.spotify.com/v1/tracks?ids={track_ids}'
        return self._get(url)

    def get_artist_genres(self, artist_ids: Union[str, List[str]]) -> dict:
        """
        Retrieves genre data for the given artist IDs.

        Args:
            artist_ids (Union[str, List[str]]): A single artist ID or a list of artist IDs.
        
        Returns:
            dict: Artist genres data.
        
        Raises:
            ValueError: If more than 50 artist IDs are provided.
        """
        if isinstance(artist_ids, list):
            if len(artist_ids) > 50:
                raise ValueError("A maximum of 50 artist IDs is allowed per request.")
            artist_ids = ','.join(artist_ids)
        url = f'https://api.spotify.com/v1/artists?ids={artist_ids}'
        return self._get(url)

    def get_song_details(self, track_id: str) -> dict:
        """
        Retrieves detailed information for a specific track, including its artists, album, genres, popularity, and explicit content.

        Args:
            track_id (str): The track ID of the song.

        Returns:
            dict: Detailed information about the track.
        """
        track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
        track_data = self._get(track_url)

        name = track_data.get("name")
        album = track_data.get("album", {}).get("name")
        artist_objs = track_data.get("artists", [])
        artists = [a.get("name") for a in artist_objs]
        popularity = track_data.get("popularity")
        explicit = track_data.get("explicit")

        artist_ids = [a.get("id") for a in artist_objs if a.get("id")]
        genres = []

        if artist_ids:
            artist_data = self.get_artist_genres(artist_ids)
            if artist_data and "artists" in artist_data:
                for artist in artist_data["artists"]:
                    genres.extend(artist.get("genres", []))

        genres = sorted(set(genres)) if genres else []

        return {
            "song_name": name,
            "artists": artists,
            "album": album,
            "genres": genres,
            "popularity": popularity,
            "explicit": explicit,
        }

    def get_top_song_ids_by_artist(self, artist_id: str, k: int = 5, market: str = 'IN') -> List[str]:
        """
        Retrieves the top k track IDs for a given artist.

        Args:
            artist_id (str): The Spotify artist ID.
            k (int): The number of top tracks to return.
            market (str): Market code (default is 'IN').

        Returns:
            List[str]: A list of top track IDs for the artist.
        """
        url = f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks?market={market}'
        data = self._get(url)
        return [track.get("id") for track in data.get("tracks", [])[:k]]


class HelperFunctions:
    @staticmethod
    def is_token_valid(access_token: str) -> bool:
        """
        Checks if a Spotify access token is still valid by calling the /v1/me endpoint.

        Parameters:
            access_token (str): The Spotify access token to check.

        Returns:
            bool: True if the token is valid, False if expired or invalid.
        """
        url = 'https://api.spotify.com/v1/me'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            print("Access token is invalid or expired.")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False

    @staticmethod
    def refresh_access_token(config_path: str, refresh_token: str) -> dict:
        """
        Refreshes Spotify access token using credentials from a config file and a stored refresh token.

        Parameters:
            config_path (str): Path to the JSON config file with client_id and client_secret.
            refresh_token (str): The stored refresh token.

        Returns:
            dict: New token data containing access_token (and possibly refresh_token).
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        client_id = config['client_id']
        client_secret = config['client_secret']

        token_url = 'https://accounts.spotify.com/api/token'
        auth_str = f"{client_id}:{client_secret}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()

        headers = {
            'Authorization': f'Basic {b64_auth}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }

        response = requests.post(token_url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to refresh token:", response.status_code, response.text)
            return None

    @staticmethod
    def nested_defaultdict_to_dict(d: Union[Dict[Any, Any], defaultdict]) -> Dict[str, Any]:
        """
        Recursively converts a (possibly nested) defaultdict into a regular dictionary
        with stringified keys to ensure compatibility with JSON serialization.

        This is especially useful when keys are non-JSON-serializable types (e.g., tuples).

        Parameters:
            d (Union[Dict[Any, Any], defaultdict]):
                The dictionary (or defaultdict) to convert.

        Returns:
            Dict[str, Any]: A standard dictionary with all keys converted to strings and
                            all nested defaultdicts replaced by dicts.
        """
        if isinstance(d, defaultdict):
            # Convert defaultdict to dict and recurse
            return {str(k): HelperFunctions.nested_defaultdict_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            # Handle nested dicts that aren't defaultdicts
            return {str(k): HelperFunctions.nested_defaultdict_to_dict(v) for k, v in d.items()}
        else:
            # Base case: return the value as-is (non-dict)
            return d

    @staticmethod
    def exponential_decay(timestamp: str, half_life_days: float = 7) -> float:
        """
        Returns a decay weight between 0 and 1 based on how old the timestamp is.
        Newer timestamps yield weights closer to 1.

        Parameters:
            timestamp (str): The timestamp to decay.
            half_life_days (float): The half-life in days to calculate the decay. Default is 7.

        Returns:
            float: Decay weight between 0 and 1.
        """
        try:
            # Try parsing with microseconds
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback to parsing without microseconds
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        days_since = (datetime.now(timezone.utc) - dt).total_seconds() / (3600 * 24)
        return math.exp(-math.log(2) * days_since / half_life_days)

    @staticmethod
    def build_user_source_profiles(track_contributions: dict, init_df: pd.DataFrame, feature_columns: list) -> dict:
        """
        Builds user source profiles by calculating weighted average vectors for each data source.

        Parameters:
            track_contributions (dict): Dictionary of track IDs mapped to a dictionary of sources and scores.
            init_df (pd.DataFrame): DataFrame with features for each track indexed by track_id.
            feature_columns (list): List of feature column names used to build the track vectors.

        Returns:
            dict: A dictionary of user source profiles where each source is mapped to its weighted average vector.
        """
        # Containers for each source
        source_weighted_sums = defaultdict(lambda: np.zeros(len(feature_columns), dtype=np.float32))
        source_total_scores = defaultdict(float)

        for track_id, sources in track_contributions.items():
            if track_id not in init_df.index:
                continue  # skip unknown tracks

            song_vector = init_df.loc[track_id, feature_columns].values.astype(np.float32)

            for source, score in sources.items():
                source_weighted_sums[source] += song_vector * score
                source_total_scores[source] += score

        # Normalize to get average vector per source
        user_source_profiles = {}
        for source in source_weighted_sums:
            total_score = source_total_scores[source]
            if total_score > 1e-6:
                user_source_profiles[source] = source_weighted_sums[source] / total_score
            else:
                user_source_profiles[source] = np.zeros(len(feature_columns), dtype=np.float32)

        return user_source_profiles

    @staticmethod
    def print_song_names_and_artists(song_ids: list, init_df: pd.DataFrame) -> None:
        """
        Prints song names and their corresponding artists from a list of song IDs.

        Parameters:
            song_ids (List[str]): List of song IDs to look up.
            init_df (pd.DataFrame): DataFrame indexed by song_id with 'name' and 'artists' columns.
        """
        for song_id in song_ids:
            if song_id in init_df.index:
                name = init_df.loc[song_id, 'name']
                artists = init_df.loc[song_id, 'artists']
                print(f"{name} by {artists}")
            else:
                print(f"{song_id} not found in dataset.")


class SpotifyDataEnricher:
    """
    Class for enriching song data by ensuring necessary columns exist, refreshing tokens when needed,
    handling API requests with retries, and updating the song data in a DataFrame.
    """

    def __init__(self, spotify, config_path: str, refresh_token: str):
        self.spotify = spotify
        self.config_path = config_path
        self.refresh_token = refresh_token

    def ensure_columns_exist(self, df: pd.DataFrame) -> None:
        """
        Ensures that specific columns ('genres', 'popularity', 'explicit') exist in the given DataFrame.
        If they do not exist, they are added with default `None` values.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to check and modify.
        """
        for col in ['genres', 'popularity', 'explicit']:
            if col not in df.columns:
                df[col] = None

    def refresh_token_if_needed(self) -> None:
        """
        Refreshes the Spotify access token if it is no longer valid. Uses the provided refresh token to
        obtain a new access token from the API.
        """
        if not HelperFunctions.is_token_valid(self.spotify.access_token):
            new_token_data = HelperFunctions.refresh_access_token(self.config_path, self.refresh_token)
            self.spotify.access_token = new_token_data['access_token']

    def update_song_row(self, df: pd.DataFrame, song_id: str, details: Optional[Dict[str, Any]]) -> None:
        """
        Updates the song information in the DataFrame for a specific song identified by its song ID.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the song data.
            song_id (str): The ID of the song to update.
            details (Optional[Dict[str, Any]]): A dictionary containing the song details to update.
        """
        if details:
            df.at[song_id, 'name'] = details.get('song_name') or df.at[song_id, 'name']
            df.at[song_id, 'artists'] = ', '.join(details.get('artists', [])) or df.at[song_id, 'artists']
            df.at[song_id, 'album_name'] = details.get('album') or df.at[song_id, 'album_name']
            df.at[song_id, 'genres'] = ', '.join(details.get('genres', []))
            df.at[song_id, 'popularity'] = details.get('popularity')
            df.at[song_id, 'explicit'] = details.get('explicit')

    def handle_request_with_retries(self, song_id: str, max_retries: int = 5, sleep_time: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Handles API requests to fetch song details with retry logic. Retries the request on failure, including
        handling rate limiting (HTTP status 429) and server errors (HTTP status codes 500, 502, 503).
        
        Parameters:
            song_id (str): The ID of the song to fetch details for.
            max_retries (int): The maximum number of retries before giving up.
            sleep_time (float): The amount of time to sleep between retries.
        
        Returns:
            Optional[Dict[str, Any]]: The song details if successful, otherwise `None`.
        """
        attempt = 0
        while attempt < max_retries:
            self.refresh_token_if_needed()

            try:
                details = self.spotify.get_song_details(song_id)
                return details  # Success

            except requests.exceptions.RequestException as e:
                response = getattr(e, 'response', None)
                if response and response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    time.sleep(retry_after)
                elif response and response.status_code in [500, 502, 503]:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(backoff)
                else:
                    return None  # Non-retryable error

            except Exception as e:
                return None

            attempt += 1
            time.sleep(sleep_time)

        return None

    def enrich_dataframe(self, df: pd.DataFrame, sleep_time: float = 0.1, max_retries: int = 5) -> pd.DataFrame:
        """
        Enriches a DataFrame of songs by fetching and updating song details through the Spotify API.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing song IDs that need to be enriched.
            sleep_time (float): Time to sleep between requests to avoid hitting rate limits.
            max_retries (int): The maximum number of retries for failed API requests.
        
        Returns:
            pd.DataFrame: The DataFrame with updated song details.
        """
        self.ensure_columns_exist(df)

        for song_id in tqdm(df.index, desc="Enriching songs"):
            details = self.handle_request_with_retries(song_id=song_id, max_retries=max_retries, sleep_time=sleep_time)
            self.update_song_row(df, song_id, details)

        return df


class SpotifyUserDataFetcher:
    """
    Class to fetch user-specific data from the Spotify API using an authenticated user session.
    """

    def __init__(self, user: 'SpotifyUserData'):
        """
        Initialize the fetcher with an authenticated Spotify user session.

        Parameters:
            user (SpotifyUserData): Authenticated Spotify user object that handles token logic and API calls.
        """
        self.user = user

    def fetch_top_tracks(self, time_range: str, limit: int) -> Optional[List[str]]:
        """
        Fetch the user's top tracks over a specified time range.

        Parameters:
            time_range (str): Time period for analysis. Options:
                              - 'short_term' (approx. last 4 weeks)
                              - 'medium_term' (approx. last 6 months)
                              - 'long_term' (years of data)
            limit (int): Number of tracks to retrieve.
        
        Returns:
            Optional[List[str]]: List of Spotify track IDs.
        """
        try:
            data = self.user.get_top_tracks(time_range=time_range, limit=limit)
            return [track['id'] for track in data['items']]
        except Exception as e:
            print(f"Error in get_top_tracks: {e}")
            return None

    def fetch_recent_tracks(self, limit: int) -> Optional[List[Tuple[str, str]]]:
        """
        Fetch the user's most recently played tracks.

        Parameters:
            limit (int): Number of recently played tracks to retrieve.
        
        Returns:
            Optional[List[Tuple[str, str]]]: List of (track_id, played_at) tuples.
        """
        try:
            data = self.user.get_recently_played(limit=limit)
            return [(item['track']['id'], item['played_at']) for item in data['items']]
        except Exception as e:
            print(f"Error in get_recent_tracks: {e}")
            return None

    def fetch_saved_tracks(self, limit: int, offset: int) -> Optional[List[Tuple[str, str]]]:
        """
        Fetch the user's saved (liked) tracks.

        Parameters:
            limit (int): Number of saved tracks to retrieve.
            offset (int): Index for pagination (e.g., 0 for first page).
        
        Returns:
            Optional[List[Tuple[str, str]]]: List of (track_id, added_at) tuples.
        """
        try:
            data = self.user.get_saved_tracks(limit=limit, offset=offset)
            return [(item['track']['id'], item['added_at']) for item in data['items']]
        except Exception as e:
            print(f"Error in get_saved_tracks: {e}")
            return None

    def fetch_top_artists(self, time_range: str, limit: int) -> Optional[List[Tuple[str, int]]]:
        """
        Fetch the user's top artists over a specified time range.

        Parameters:
            time_range (str): Time period for analysis. Options:
                              - 'short_term' (approx. last 4 weeks)
                              - 'medium_term' (approx. last 6 months)
                              - 'long_term' (years of data)
            limit (int): Number of top artists to retrieve.
        
        Returns:
            Optional[List[Tuple[str, int]]]: List of (artist_id, popularity) tuples.
        """
        try:
            data = self.user.get_top_artists(time_range=time_range, limit=limit)
            return [(artist['id'], artist['popularity']) for artist in data['items']]
        except Exception as e:
            print(f"Error in get_top_artists: {e}")
            return None
    
    def rank_user_tracks(
        self,
        top_tracks_time_range: str,
        top_tracks_limit: int,
        recent_tracks_limit: int,
        saved_tracks_limit: int,
        saved_tracks_offset: int,
        top_artists_time_range: str,
        top_artists_limit: int,
        source_weights: Dict[str, int]
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, float]]]:
        """
        Combines and ranks tracks from multiple sources using a configurable weighted system.
        Applies exponential time decay to recent and saved tracks to prioritize newer items.
        
        Parameters:
            top_tracks_time_range (str): Time range for top tracks (e.g., 'medium_term').
            top_tracks_limit (int): Limit for number of top tracks.
            recent_tracks_limit (int): Limit for number of recently played tracks.
            saved_tracks_limit (int): Limit for number of saved tracks.
            saved_tracks_offset (int): Pagination offset for saved tracks.
            top_artists_time_range (str): Time range for top artists (e.g., 'medium_term').
            top_artists_limit (int): Limit for number of top artists.
            source_weights (Dict[str, int]): Weights for each data source (top tracks, recent tracks, etc.).
        
        Returns:
            Tuple[List[Tuple[str, float]], Dict[str, Dict[str, float]]]: A tuple with:
            - ranked_tracks: List of tracks ranked by score.
            - track_contributions: Contributions for each track from different sources.
        """
        track_contributions = defaultdict(lambda: defaultdict(float))

        # 1. Top Tracks
        if 'top_tracks' in source_weights:
            top_track_ids = self.fetch_top_tracks(top_tracks_time_range, top_tracks_limit)
            for tid in top_track_ids:
                track_contributions[tid]['top_tracks'] += source_weights['top_tracks']

        # 2. Recently Played (with time decay)
        if 'recent_tracks' in source_weights:
            recent_track_items = self.fetch_recent_tracks(recent_tracks_limit)
            for tid, timestamp in recent_track_items:  # should return (track_id, played_at)
                decay = HelperFunctions.exponential_decay(timestamp)
                track_contributions[tid]['recent_tracks'] += source_weights['recent_tracks'] * decay

        # 3. Saved Tracks (with time decay)
        if 'saved_tracks' in source_weights:
            saved_track_items = self.fetch_saved_tracks(saved_tracks_limit, saved_tracks_offset)
            for tid, timestamp in saved_track_items:  # should return (track_id, added_at)
                decay = HelperFunctions.exponential_decay(timestamp)
                track_contributions[tid]['saved_tracks'] += source_weights['saved_tracks'] * decay

        # 4. Tracks by Top Artists
        if 'top_artist_tracks' in source_weights:
            top_artist_ids = self.fetch_top_artists(top_artists_time_range, top_artists_limit)
            for artist_id, _ in top_artist_ids:
                artist_track_ids = self.user.get_top_song_ids_by_artist(artist_id)
                for tid in artist_track_ids:
                    track_contributions[tid]['top_artist_tracks'] += source_weights['top_artist_tracks']

        # Aggregate total score
        track_scores = {
            tid: sum(source_dict.values())
            for tid, source_dict in track_contributions.items()
        }

        # Sort by descending score
        ranked_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_track_ids = [(track_id, score) for track_id, score in ranked_tracks]

        return ranked_track_ids, track_contributions


class EpsilonGreedyBandit:
    """
    A class implementing the epsilon-greedy strategy for action selection in a multi-armed bandit problem.
    The class manages the exploration-exploitation trade-off by selecting an action based on an epsilon probability,
    where exploration picks a random action, and exploitation picks the best-known action based on weighted feedback.
    """
    
    def __init__(self, sources: List[str], epsilon: float = 0.1, initial_weights: Optional[Dict[str, float]] = None):
        self.sources = sources
        self.epsilon = epsilon
        if initial_weights:
            self.source_weights = initial_weights
        else:
            self.source_weights = {source: 1.0 for source in sources}
        self.action_counts = {source: 0 for source in sources}

    def select_action(self) -> str:
        """
        Select an action (source) based on epsilon-greedy strategy.
        Explores by selecting a random source with probability epsilon, and exploits by selecting the source with the highest weight.
        
        Returns:
            str: The selected action (source).
        """
        if random.random() < self.epsilon:
            # Exploration: Select a random source
            selected_source = random.choice(self.sources)
        else:
            # Exploitation: Select the source with the highest weight
            selected_source = max(self.source_weights, key=self.source_weights.get)
        
        return selected_source
    
    def update_source_weights(self, selected_source: str, feedback_score: float) -> float:
        """
        Update the weight of the selected source based on the feedback score.
        The weight is updated using a learning rate and feedback score, followed by normalization of all weights.
        
        Args:
            selected_source (str): The source that was selected.
            feedback_score (float): The feedback score received for the selected source.
        
        Returns:
            float: The previous weight of the selected source before the update.
        """
        learning_rate = 0.1
        previous_weight = self.source_weights[selected_source]

        updated_weight = previous_weight + learning_rate * (feedback_score - previous_weight)
        self.source_weights[selected_source] = updated_weight

        self._normalize_weights()

        return previous_weight

    def _normalize_weights(self) -> None:
        """
        Normalize the weights of all sources so that they sum up to 1.
        This ensures the weights remain proportional and maintain a consistent scale.
        """
        total = sum(self.source_weights.values())
        if total > 0:
            self.source_weights = {
                source: weight / total for source, weight in self.source_weights.items()
            }


class RecommendationFeedbackCollector:
    """
    A class responsible for collecting user feedback on recommended songs from multiple sources.
    It retrieves song recommendations for each source, prompts the user to rate the songs, 
    and collects feedback to be used for updating the recommendation system.
    """
    
    def __init__(self, user_source_profile: Dict[str, List[float]], index: faiss.Index, song_ids: List[str], 
                 init_df: pd.DataFrame, print_song_names_and_artists: callable) -> None:
        self.user_source_profile = user_source_profile
        self.index = index
        self.song_ids = song_ids
        self.init_df = init_df
        self.print_song_names_and_artists = print_song_names_and_artists
        self.source_recommendations = {}
        self.user_feedback = {}

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize the input vector using L2 normalization for similarity search in FAISS.
        
        Args:
            vector (List[float]): The vector to be normalized.
        
        Returns:
            List[float]: The normalized vector.
        """
        normalized_vector = vector.reshape(1, -1).copy()
        faiss.normalize_L2(normalized_vector)
        return normalized_vector

    def _get_recommendations_for_source(self, source: str, vector: List[float]) -> Tuple[List[str], List[float]]:
        """
        Retrieve the top song recommendations for a given source by searching the FAISS index for similar songs.
        
        Args:
            source (str): The source for which recommendations are being fetched.
            vector (List[float]): The vector representing the user's profile for the source.
        
        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of recommended song IDs and their corresponding scores.
        """
        normalized_vector = self._normalize_vector(vector)
        D, I = self.index.search(normalized_vector, k=10)
        recommended_ids = [self.song_ids[i] for i in I[0]]
        return recommended_ids, D[0].tolist()

    def _get_song_metadata(self, song_id: str) -> Tuple[str, str]:
        """
        Fetch metadata (name and artist) for a given song ID.
        
        Args:
            song_id (str): The ID of the song.
        
        Returns:
            Tuple[str, str]: The name and artist(s) of the song.
        """
        try:
            row = self.init_df.loc[song_id]
            song_name = row['name']
            artist_name = ', '.join(row['artists']) if isinstance(row['artists'], list) else row['artists']
        except KeyError:
            song_name = "Unknown"
            artist_name = "Unknown"
        return song_name, artist_name

    def _ask_for_ratings(self, song_ids: List[str]) -> List[Optional[int]]:
        """
        Prompt the user to rate a list of songs, and return the ratings.
        
        Args:
            song_ids (List[str]): A list of song IDs to be rated by the user.
        
        Returns:
            List[Optional[int]]: A list of ratings (integers between 0 and 10, or None for skipped songs).
        """
        ratings = []
        for song_id in song_ids:
            song_name, artist_name = self._get_song_metadata(song_id)
            while True:
                try:
                    user_input = input(f"Rate '{song_name}' by {artist_name} (0-10, 'skip', or 'exit'): ").strip().lower()
                    if user_input == "exit":
                        print("Exiting rating input early.")
                        return ratings
                    if user_input == "skip":
                        print(f"Skipping '{song_name}' by {artist_name}.")
                        ratings.append(None)  # or you can skip appending entirely if you prefer
                        break
                    rating = int(user_input)
                    if 0 <= rating <= 10:
                        ratings.append(rating)
                        break
                    else:
                        print("Please enter a valid rating between 0 and 10.")
                except ValueError:
                    print("Invalid input, please enter an integer, 'skip', or 'exit'.")
        return ratings

    def collect_feedback(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[Optional[int]]]]:
        """
        Collect feedback from the user for song recommendations from multiple sources.
        For each source, the user is asked to rate a set of recommended songs, 
        and the ratings are returned alongside the recommendations.
        
        Returns:
            Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[Optional[int]]]]:
                - source_recommendations: A dictionary with source names as keys, containing song recommendations and their scores.
                - user_feedback: A dictionary with source names as keys, containing user feedback ratings for each song.
        """
        for source, vector in self.user_source_profile.items():
            recommended_ids, scores = self._get_recommendations_for_source(source, vector)

            self.source_recommendations[source] = {
                'song_ids': recommended_ids,
                'scores': scores
            }

            print(f"\nSource: {source}")
            print("Recommended song IDs:", recommended_ids)
            print("Scores:", scores)
            self.print_song_names_and_artists(recommended_ids, self.init_df)

            print("\nPlease rate the recommendations (0 to 10) for each song:")
            ratings = self._ask_for_ratings(recommended_ids)
            self.user_feedback[source] = ratings
            print("-" * 40)

        return self.source_recommendations, self.user_feedback


# class SpotifyUserData:
#     """
#     SpotifyUserData Class - Methods Summary

#     | Method                | Description                                        | Endpoint                                      |
#     |-----------------------|----------------------------------------------------|-----------------------------------------------|
#     | get_user_profile      | Fetches user's profile info                        | /v1/me                                        |
#     | get_recently_played   | Recently played tracks (last 24h)                  | /v1/me/player/recently-played                 |
#     | get_top_tracks        | User's top tracks over specified time range        | /v1/me/top/tracks                             |
#     | get_saved_tracks      | Tracks saved (liked) by the user                   | /v1/me/tracks                                 |
#     | get_top_artists       | User's top artists over specified time range       | /v1/me/top/artists                            |
#     | get_audio_features    | Audio features (tempo, energy, etc.) for track IDs | /v1/audio-features                            |
#     | get_track_metadata    | Metadata for one or multiple track IDs             | /v1/tracks                                    |
#     | get_artist_genres     | Genres for one or more artist IDs                  | /v1/artists                                   |
#     | get_song_details      | All-in-one: metadata + artist genres for a track   | /v1/tracks + /v1/artists                      |
#     """
#     def __init__(self, access_token):
#         self._access_token = access_token
#         self._headers = {
#             'Authorization': f'Bearer {self._access_token}'
#         }

#     @property
#     def access_token(self):
#         return self._access_token

#     @access_token.setter
#     def access_token(self, token):
#         self._access_token = token

#     def _get_headers(self):
#         return {'Authorization': f'Bearer {self._access_token}'}

#     def _get(self, url: str) -> dict:
#         while True:
#             response = requests.get(url, headers=self._get_headers())
#             if response.status_code == 429:
#                 retry_after = int(response.headers.get("Retry-After", 1))
#                 print(f"Rate limit hit. Retrying after {retry_after} seconds...")
#                 time.sleep(retry_after)
#                 continue
#             elif response.status_code == 401:
#                 print("Access token expired or unauthorized. Refreshing token...")
#                 # Refresh the token here (you need to implement token refresher logic as per your needs)
#                 continue
#             elif response.status_code != 200:
#                 raise requests.exceptions.HTTPError(
#                     f"Spotify API error {response.status_code}: {response.text}"
#                 )
#             return response.json()

#     def get_user_profile(self) -> dict:
#         return self._get('https://api.spotify.com/v1/me')

#     def get_recently_played(self, limit: int = 20) -> dict:
#         url = f'https://api.spotify.com/v1/me/player/recently-played?limit={limit}'
#         return self._get(url)

#     def get_top_tracks(self, time_range: str = 'medium_term', limit: int = 20) -> dict:
#         url = f'https://api.spotify.com/v1/me/top/tracks?limit={limit}&time_range={time_range}'
#         return self._get(url)

#     def get_saved_tracks(self, limit: int = 20, offset: int = 0) -> dict:
#         url = f'https://api.spotify.com/v1/me/tracks?limit={limit}&offset={offset}'
#         return self._get(url)

#     def get_top_artists(self, time_range: str = 'medium_term', limit: int = 20) -> dict:
#         url = f'https://api.spotify.com/v1/me/top/artists?limit={limit}&time_range={time_range}'
#         return self._get(url)

#     def get_audio_features(self, track_ids: Union[str, List[str]]) -> dict:
#         if isinstance(track_ids, list):
#             if len(track_ids) > 100:
#                 raise ValueError("A maximum of 100 track IDs is allowed per request.")
#             track_ids = ','.join(track_ids)
#         url = f'https://api.spotify.com/v1/audio-features?ids={track_ids}'
#         return self._get(url)

#     def get_track_metadata(self, track_ids: Union[str, List[str]]) -> dict:
#         if isinstance(track_ids, list):
#             if len(track_ids) > 50:
#                 raise ValueError("A maximum of 50 track IDs is allowed per request.")
#             track_ids = ','.join(track_ids)
#         url = f'https://api.spotify.com/v1/tracks?ids={track_ids}'
#         return self._get(url)

#     def get_artist_genres(self, artist_ids: Union[str, List[str]]) -> dict:
#         if isinstance(artist_ids, list):
#             if len(artist_ids) > 50:
#                 raise ValueError("A maximum of 50 artist IDs is allowed per request.")
#             artist_ids = ','.join(artist_ids)
#         url = f'https://api.spotify.com/v1/artists?ids={artist_ids}'
#         return self._get(url)

#     def get_song_details(self, track_id: str) -> dict:
#         track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
#         track_data = self._get(track_url)

#         name = track_data.get("name")
#         album = track_data.get("album", {}).get("name")
#         artist_objs = track_data.get("artists", [])
#         artists = [a.get("name") for a in artist_objs]
#         popularity = track_data.get("popularity")
#         explicit = track_data.get("explicit")

#         artist_ids = [a.get("id") for a in artist_objs if a.get("id")]
#         genres = []

#         if artist_ids:
#             artist_data = self.get_artist_genres(artist_ids)
#             if artist_data and "artists" in artist_data:
#                 for artist in artist_data["artists"]:
#                     genres.extend(artist.get("genres", []))

#         genres = sorted(set(genres)) if genres else []

#         return {
#             "song_name": name,
#             "artists": artists,
#             "album": album,
#             "genres": genres,
#             "popularity": popularity,
#             "explicit": explicit,
#         }
