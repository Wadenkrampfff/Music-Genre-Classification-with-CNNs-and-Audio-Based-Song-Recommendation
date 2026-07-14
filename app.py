from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent

BANNER_PATH = PROJECT_ROOT / "images" / "banner.png"

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "final_cnn_genre_classifier_augmented.pth"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "results"
    / "embeddings"
    / "song_embeddings.npz"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gtzan"
    / "Data"
    / "genres_original"
)


# ============================================================
# Final CNN Architecture
# ============================================================

class FinalCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    classes = np.asarray(checkpoint["classes"])

    model = FinalCNN(
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    metadata = {
        "classes": classes,
        "x_mean": float(checkpoint["x_mean"]),
        "x_std": float(checkpoint["x_std"]),
        "n_mels": int(checkpoint["n_mels"]),
        "target_frames": int(checkpoint["target_frames"]),
        "sample_rate": int(checkpoint["sample_rate"]),
    }

    return model, metadata, device

@st.cache_data
def load_embedding_database():
    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Embedding database not found:\n{EMBEDDING_PATH}"
        )

    database = np.load(
        EMBEDDING_PATH,
        allow_pickle=True,
    )

    embeddings = np.asarray(
        database["embeddings"],
        dtype=np.float32,
    )

    song_names = np.asarray(
        database["song_names"],
        dtype=str,
    )

    song_genres = np.asarray(
        database["song_genres"],
        dtype=str,
    )

    if len(embeddings) != len(song_names):
        raise ValueError(
            "The number of embeddings and song names does not match."
        )

    if len(embeddings) != len(song_genres):
        raise ValueError(
            "The number of embeddings and genres does not match."
        )

    return embeddings, song_names, song_genres


# ============================================================
# Audio Preprocessing
# ============================================================

def compute_mel_spectrogram(
    file_path: Path,
    sample_rate: int,
    n_mels: int,
    target_frames: int,
) -> np.ndarray:
    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(
            "The uploaded audio file is empty."
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    if mel_db.shape[1] >= target_frames:
        mel_db = mel_db[:, :target_frames]

    else:
        pad_width = (
            target_frames - mel_db.shape[1]
        )

        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
        )

    return mel_db


# ============================================================
# Audio Classification
# ============================================================

def predict_uploaded_audio(
    uploaded_file,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    model, metadata, device = load_model()

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(
            uploaded_file.getbuffer()
        )

        temporary_path = Path(
            temporary_file.name
        )

    try:
        mel = compute_mel_spectrogram(
            temporary_path,
            sample_rate=metadata["sample_rate"],
            n_mels=metadata["n_mels"],
            target_frames=metadata["target_frames"],
        )

        normalized_mel = (
            mel - metadata["x_mean"]
        ) / metadata["x_std"]

        model_input = normalized_mel[
            np.newaxis,
            np.newaxis,
            :,
            :,
        ]

        input_tensor = torch.tensor(
            model_input,
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            ).cpu().numpy()[0]

        results = pd.DataFrame({
            "Genre": metadata["classes"],
            "Probability": probabilities,
        }).sort_values(
            by="Probability",
            ascending=False,
        )

        return (
            results,
            mel,
            metadata["sample_rate"],
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

def extract_uploaded_embedding(
    uploaded_file,
) -> tuple[np.ndarray, np.ndarray, int]:
    model, metadata, device = load_model()

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(
            uploaded_file.getbuffer()
        )

        temporary_path = Path(
            temporary_file.name
        )

    try:
        mel = compute_mel_spectrogram(
            temporary_path,
            sample_rate=metadata["sample_rate"],
            n_mels=metadata["n_mels"],
            target_frames=metadata["target_frames"],
        )

        normalized_mel = (
            mel - metadata["x_mean"]
        ) / metadata["x_std"]

        model_input = normalized_mel[
            np.newaxis,
            np.newaxis,
            :,
            :,
        ]

        input_tensor = torch.tensor(
            model_input,
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            features = model.features(
                input_tensor
            )

            features = torch.flatten(
                features,
                start_dim=1,
            )

            embedding = model.classifier[1](
                features
            )

            embedding = model.classifier[2](
                embedding
            )

        embedding = (
            embedding
            .cpu()
            .numpy()[0]
        )

        return (
            embedding,
            mel,
            metadata["sample_rate"],
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

def recommend_uploaded_audio(
    uploaded_file,
    top_n: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    (
        database_embeddings,
        song_names,
        song_genres,
    ) = load_embedding_database()

    (
        query_embedding,
        mel,
        sample_rate,
    ) = extract_uploaded_embedding(
        uploaded_file
    )

    if (
        query_embedding.shape[0]
        != database_embeddings.shape[1]
    ):
        raise ValueError(
            "The uploaded-song embedding and database "
            "embeddings have different dimensions."
        )

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        database_embeddings,
    )[0]

    # Prevent a GTZAN song from recommending itself
    same_name_mask = (
        song_names == uploaded_file.name
    )

    similarities[
        same_name_mask
    ] = -np.inf

    top_indices = np.argsort(
        similarities
    )[::-1][:top_n]

    recommendations = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        genre = song_genres[index]
        song_name = song_names[index]

        song_path = (
            DATASET_PATH
            / genre
            / song_name
        )

        recommendations.append({
            "Rank": rank,
            "Song": song_name,
            "Genre": genre,
            "Similarity": float(
                similarities[index]
            ),
            "Path": song_path,
        })

    recommendation_df = pd.DataFrame(
        recommendations
    )

    return (
        recommendation_df,
        mel,
        sample_rate,
    )

# ============================================================
# Mel-Spectrogram Plot
# ============================================================

def create_mel_figure(
    mel: np.ndarray,
    sample_rate: int,
):
    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    image = librosa.display.specshow(
        mel,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axis,
    )

    figure.colorbar(
        image,
        ax=axis,
        format="%+2.0f dB",
    )

    axis.set_title(
        "Generated Mel-Spectrogram"
    )

    axis.set_xlabel("Time")
    axis.set_ylabel("Mel Frequency")

    figure.tight_layout()

    return figure


# ============================================================
# Page Header
# ============================================================

if BANNER_PATH.exists():
    st.image(
        str(BANNER_PATH),
        use_container_width=True,
    )

st.title(
    "Music Genre Classification & Recommendation"
)

st.write(
    """
    Upload an audio file to predict its music genre and retrieve
    acoustically similar songs using embeddings from the final CNN.
    """
)


classification_tab, recommendation_tab = st.tabs(
    [
        "Genre Classification",
        "Song Recommendation",
    ]
)


# ============================================================
# Genre Classification Tab
# ============================================================

with classification_tab:
    st.header("Genre Classification")

    st.write(
        """
        The uploaded audio file is converted into a Mel-spectrogram.
        This representation is then processed by the final CNN trained
        on the complete GTZAN dataset.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a WAV or MP3 file",
        type=["wav", "mp3"],
        key="classification_upload",
    )

    if uploaded_file is not None:
        st.subheader("1. Uploaded Audio")

        st.audio(uploaded_file)

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

        with st.spinner(
            "Processing and classifying audio..."
        ):
            try:
                (
                    prediction_df,
                    mel,
                    sample_rate,
                ) = predict_uploaded_audio(
                    uploaded_file
                )

                predicted_genre = str(
                    prediction_df.iloc[0]["Genre"]
                ).title()

                confidence = float(
                    prediction_df.iloc[0][
                        "Probability"
                    ]
                )

                # ------------------------------------------------
                # Mel-Spectrogram
                # ------------------------------------------------

                st.subheader(
                    "2. Generated Mel-Spectrogram"
                )

                st.caption(
                    "This is the image-like audio representation "
                    "that is passed into the CNN."
                )

                mel_figure = create_mel_figure(
                    mel=mel,
                    sample_rate=sample_rate,
                )

                st.pyplot(
                    mel_figure,
                    use_container_width=True,
                )

                plt.close(mel_figure)

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                st.subheader(
                    "3. Genre Prediction"
                )

                metric_column, explanation_column = (
                    st.columns([1, 2])
                )

                with metric_column:
                    st.metric(
                        label="Predicted Genre",
                        value=predicted_genre,
                    )

                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1%}",
                    )

                with explanation_column:
                    st.write(
                        """
                        The CNN produces one probability for each of
                        the ten GTZAN genres. The class with the highest
                        probability is selected as the prediction.
                        """
                    )

                    if confidence < 0.50:
                        st.warning(
                            "The model is uncertain about this prediction."
                        )

                    elif confidence < 0.75:
                        st.info(
                            "The model has moderate confidence in this prediction."
                        )

                    else:
                        st.success(
                            "The model has high confidence in this prediction."
                        )

                # ------------------------------------------------
                # Probability Distribution
                # ------------------------------------------------

                st.subheader(
                    "4. Genre Probability Distribution"
                )

                chart_df = (
                    prediction_df
                    .set_index("Genre")[["Probability"]]
                )

                st.bar_chart(
                    chart_df
                )

                with st.expander(
                    "Show all probability values"
                ):
                    display_df = (
                        prediction_df.copy()
                    )

                    display_df[
                        "Probability"
                    ] = (
                        display_df[
                            "Probability"
                        ]
                        * 100
                    ).round(2)

                    display_df = (
                        display_df.rename(
                            columns={
                                "Probability":
                                "Probability (%)"
                            }
                        )
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    """
                    Predictions for songs outside the GTZAN dataset
                    may be unreliable because the model was trained
                    on a relatively small and historic genre dataset.
                    """
                )

            except Exception as error:
                st.error(
                    "The audio file could not be processed."
                )

                st.exception(error)


# ============================================================
# Recommendation Tab
# ============================================================

with recommendation_tab:
    st.header("Song Recommendation")

    st.write(
        """
        Upload a song to generate a 256-dimensional CNN embedding.
        The embedding is compared with the precomputed GTZAN song
        database using cosine similarity.
        """
    )

    recommendation_file = st.file_uploader(
        "Upload a WAV or MP3 file",
        type=["wav", "mp3"],
        key="recommendation_upload",
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    if recommendation_file is not None:
        st.subheader("1. Query Song")

        st.audio(
            recommendation_file
        )

        st.success(
            f"Loaded: {recommendation_file.name}"
        )

        with st.spinner(
            "Extracting CNN embedding and searching for similar songs..."
        ):
            try:
                (
                    recommendations,
                    recommendation_mel,
                    recommendation_sample_rate,
                ) = recommend_uploaded_audio(
                    uploaded_file=recommendation_file,
                    top_n=top_n,
                )

                st.subheader(
                    "2. Query Mel-Spectrogram"
                )

                recommendation_figure = (
                    create_mel_figure(
                        mel=recommendation_mel,
                        sample_rate=(
                            recommendation_sample_rate
                        ),
                    )
                )

                st.pyplot(
                    recommendation_figure,
                    use_container_width=True,
                )

                plt.close(
                    recommendation_figure
                )

                st.subheader(
                    f"3. Top {top_n} Recommendations"
                )

                st.caption(
                    """
                    A high cosine similarity means that two songs
                    occupy nearby positions in the learned CNN
                    embedding space.
                    """
                )

                table_df = recommendations[
                    [
                        "Rank",
                        "Song",
                        "Genre",
                        "Similarity",
                    ]
                ].copy()

                table_df[
                    "Genre"
                ] = (
                    table_df["Genre"]
                    .str.title()
                )

                table_df[
                    "Similarity"
                ] = (
                    table_df["Similarity"]
                    * 100
                ).round(2)

                table_df = table_df.rename(
                    columns={
                        "Similarity":
                        "Similarity (%)"
                    }
                )

                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "4. Listen to Recommendations"
                )

                for _, row in recommendations.iterrows():
                    song_path = Path(
                        row["Path"]
                    )

                    with st.container(
                        border=True
                    ):
                        title_column, score_column = (
                            st.columns([3, 1])
                        )

                        with title_column:
                            st.markdown(
                                f"### {int(row['Rank'])}. "
                                f"{row['Song']}"
                            )

                            st.write(
                                "Genre: "
                                f"**{str(row['Genre']).title()}**"
                            )

                        with score_column:
                            st.metric(
                                label="Similarity",
                                value=(
                                    f"{float(row['Similarity']):.1%}"
                                ),
                            )

                        if song_path.exists():
                            st.audio(
                                str(song_path)
                            )

                        else:
                            st.warning(
                                "The recommendation was found, "
                                "but the corresponding audio file "
                                "is not available locally."
                            )

                st.info(
                    """
                    Cross-genre recommendations are possible and
                    expected. The system searches for similar CNN
                    embeddings rather than enforcing identical
                    genre labels.
                    """
                )

            except Exception as error:
                st.error(
                    "Recommendations could not be generated."
                )

                st.exception(error)