# NOTE: The code in this file comes from a Google Colab notebook.
# Hence, the structure might be a bit weird, as not everything is
# executed at the same time.

# This file contains the code that is used to download the trailers
# from YouTube and save their audio abd video data to a google drive.

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive/')

import os
path = "/content/drive/MyDrive/trailer_classification"
os.chdir(path)

# Download the necessary libraries for the data pipleline

import asyncio
import os
import sys
import json
import logging
import math
import time
from typing import List, Optional
from dataclasses import dataclass, asdict

from pytube import YouTube
from pytube.exceptions import PytubeError, VideoUnavailable
import pandas as pd

# Set hardcoded settings
YOUTUBE_BASE_URL = "https://youtube.com/watch?v={}"
DATA_DIR = "data"
DOWNLOADS_DIR = "downloads"

AUDIO_DIR = "audio"
VIDEO_DIR = "video"

AUDIO_DL_PATH = os.path.join(DOWNLOADS_DIR, AUDIO_DIR)
VIDEO_DL_PATH = os.path.join(DOWNLOADS_DIR, VIDEO_DIR)

TRAILER_INFO_FILENAME = "trailer_dl_info.csv"
trailer_info_path = os.path.join(DATA_DIR, TRAILER_INFO_FILENAME)


@dataclass
class TrailerDownloadOptions:
    """Options needed to download a trailer."""

    index: int
    video_id: str
    url: str
    res: str
    audio_downloaded: Optional[bool] = None
    video_downloaded: Optional[bool] = None
    audio_error_msg: Optional[str] = None
    video_error_msg: Optional[str] = None
    audio_is_available: Optional[bool] = None
    video_is_available: Optional[bool] = None

    def __post_init__(self):
        def _parse_bool(val):
            if val is not None:
                if math.isnan(val):
                    return None
                return val

        def _parse_error_msg(val):
            if val is not None:
                if not isinstance(val, str) and math.isnan(val):
                    return None
                return val

        self.audio_downloaded = _parse_bool(self.audio_downloaded)
        self.video_downloaded = _parse_bool(self.video_downloaded)
        self.audio_is_available = _parse_bool(self.audio_is_available)
        self.video_is_available = _parse_bool(self.video_is_available)
        self.audio_error_msg = _parse_error_msg(self.audio_error_msg)
        self.video_error_msg = _parse_error_msg(self.video_error_msg)

    @classmethod
    def parse(cls, index: int, data: dict):
        """Parse relevant options from raw input."""
        yt_data = data["youtube"]["trailers"][0]
        yt_id = yt_data["id"]
        url = YOUTUBE_BASE_URL.format(yt_id)
        res = f"{yt_data['dims'][1]}p"
        return cls(index, yt_id, url, res)

    def as_dict(self) -> dict:
        """Parse object to dict"""
        return asdict(self)

    def download_status(self, audio_or_video: str):
        if audio_or_video == "both":
            if self.audio_downloaded is None or self.video_downloaded is None:
                return None
            return self.audio_downloaded and self.video_downloaded
        elif audio_or_video == "video":
            return self.video_downloaded
        elif audio_or_video == "audio":
            return self.audio_downloaded


def load_metadata(path: str = "data/metadata.json"):
    """Loads metadata file from path"""
    logger = logging.getLogger(__name__)
    logger.info("Opening metadata file at location %s.", path)
    root_key = "trailers12k"
    try:
        with open(path, "r", encoding="utf-8") as file:
            metadata = json.load(file)[root_key]
    except FileNotFoundError:
        logger.exception("An error occured while loading %s.", path)
    else:
        logger.info("Sucessfully loaded %s.", path)
        return metadata


async def check_avail(
    trailer_dl_options: TrailerDownloadOptions,
    audio_or_video: str = "both",
    audio_dl_dir=AUDIO_DL_PATH,
    video_dl_dir=VIDEO_DL_PATH,
    fallback_res: str = "360p",
):
    """Check if YouTube content is available."""
    logger = logging.getLogger(__name__)

    check_options = {"both", "audio", "video"}
    if audio_or_video not in check_options:
        raise ValueError(f"Argument audio_or_video needs to be in {check_options}.")

    logger.info(
        "Checking availability of %s, for audio_or_video: %s.",
        trailer_dl_options,
        audio_or_video,
    )

    check_video = audio_or_video in ("both", "video")
    check_audio = audio_or_video in ("both", "audio")

    async def check_audio_avail(youtube):
        logger.info("Checking audio availibility of %s", trailer_dl_options.index)
        await asyncio.sleep(0.0001)
        try:
            streams = youtube.streams.filter(only_audio=True)
            if streams:
                logger.info(
                    "Audio for trailer %s is available!", trailer_dl_options.index
                )
                trailer_dl_options.audio_is_available = True

            else:
                logger.warning(
                    "Audio for trailer %s is unavailable!", trailer_dl_options.index
                )
                trailer_dl_options.audio_is_available = False

        except (PytubeError, KeyError):
            logger.exception(
                "An error occured when check availibility of audio of %s",
                trailer_dl_options.index,
            )
            trailer_dl_options.audio_is_available = False

    async def check_video_avail(youtube):
        logger.info("Checking video availibility of %s", trailer_dl_options.index)
        await asyncio.sleep(0.0001)
        try:
            streams = youtube.streams.filter(
                res=trailer_dl_options.res, only_video=True
            )
            if streams:
                trailer_dl_options.video_is_available = True
                logger.info(
                    "Video for trailer %s is available!", trailer_dl_options.index
                )
            else:
                streams = youtube.streams.filter(res=fallback_res, only_video=True)
                if not streams:
                    logger.warning(
                        "Video for trailer %s is unavailable!", trailer_dl_options.index
                    )
                    trailer_dl_options.video_is_available = False
                else:
                    logger.info(
                        "Found video with backup res for trailer %s", trailer_dl_options
                    )
                    trailer_dl_options.video_is_available = True

        except (PytubeError, KeyError):
            logger.exception(
                "An error occured when check availibility of video of %s",
                trailer_dl_options.index,
            )
            trailer_dl_options.video_is_available = False

    youtube = YouTube(trailer_dl_options.url)
    to_check = []
    if check_audio:
        to_check.append((check_audio_avail, youtube))
    if check_video:
        to_check.append((check_video_avail, youtube))

    await asyncio.gather(*[func(arg) for func, arg in to_check])


def download_yt(
    trailer_dl_options: TrailerDownloadOptions,
    audio_or_video: str = "both",
    audio_dl_dir=AUDIO_DL_PATH,
    video_dl_dir=VIDEO_DL_PATH,
    fallback_res: str = "360p",
):
    """Download YouTube content."""
    logger = logging.getLogger(__name__)

    download_options = {"both", "audio", "video"}
    if audio_or_video not in download_options:
        raise ValueError(f"Argument download needs to be in {download_options}.")

    logger.info(
        "Starting download with %s, audio_or_video: %s.",
        trailer_dl_options,
        audio_or_video,
    )

    download_video = audio_or_video in ("both", "video")
    download_audio = audio_or_video in ("both", "audio")

    youtube = YouTube(trailer_dl_options.url)
    if download_audio:
        try:
            logger.info(
                "Starting download of audio: %s, %s",
                trailer_dl_options.index,
                youtube.title,
            )
            # Filter for only audio stream, and download
            youtube.streams.filter(only_audio=True).first().download(
                audio_dl_dir, filename=f"{trailer_dl_options.index}.mp3"
            )
            print(youtube.streams.filter(only_audio=True))
        except (PytubeError, KeyError) as exception:
            logger.exception("Couldn't download trailer: %s", trailer_dl_options.index)
            trailer_dl_options.audio_downloaded = False
            trailer_dl_options.audio_error_msg = str(exception)
        else:
            trailer_dl_options.audio_downloaded = True
            logger.info(
                "Done downloading audio: %s, %s.",
                trailer_dl_options.index,
                youtube.title,
            )
    if download_video:
        try:
            logger.info(
                "Starting download of video of file %s, %s",
                trailer_dl_options.index,
                youtube.title,
            )
            avail_streams = youtube.streams.filter(
                res=trailer_dl_options.res, only_video=True
            )
            if avail_streams:
                avail_streams.first().download(
                    video_dl_dir, filename=f"{trailer_dl_options.index}.mp4"
                )
            else:
                logger.warning(
                    "Couldn't find video file with res %s, trailer %s, %s",
                    trailer_dl_options.res,
                    trailer_dl_options.index,
                    youtube.title,
                )
                logger.warning("Available streams: %s", youtube.streams)
                logger.warning("Attempting to download fallback res %s", fallback_res)

                backup_stream = youtube.streams.filter(
                    res=fallback_res, only_video=True
                )
                if not backup_stream:
                    raise VideoUnavailable(trailer_dl_options.video_id)
                backup_stream.first().download(
                    video_dl_dir, filename=f"{trailer_dl_options.index}.mp4"
                )
        except (PytubeError, KeyError) as exception:
            trailer_dl_options.video_downloaded = False
            trailer_dl_options.video_error_msg = str(exception)
            logger.exception("Couldn't download trailer: %s", trailer_dl_options.index)
        else:
            trailer_dl_options.video_downloaded = True
            logger.info(
                "Done downloading video: %s, %s.",
                trailer_dl_options.index,
                youtube.title,
            )


def parse_trailer_dl_options(
    metadata_path: str = "data/metadata.json",
) -> List[TrailerDownloadOptions]:
    """Parse trailer download options from metadata."""
    logger = logging.getLogger(__name__)
    metadata = load_metadata(path=metadata_path)
    logger.info("Parsing metadata...")
    trailer_dl_options = [
        TrailerDownloadOptions.parse(idx, trailer_data)
        for idx, trailer_data in metadata.items()
    ]
    logger.info("Done parsing trailer download options from metadata.")
    return trailer_dl_options


def save_trailer_dl_options(
    trailer_dl_options: List[TrailerDownloadOptions], path="data/trailer_dl_info.csv"
):
    """Save trailer download options to a csv file."""
    logger = logging.getLogger(__name__)
    logger.info("Saving trailer download options to %s...", path)
    data = pd.DataFrame(
        [trailer_dl_option.as_dict() for trailer_dl_option in trailer_dl_options]
    )
    data.set_index("index", inplace=True)
    data.to_csv(path)


def load_trailer_dl_info(
    path: str = "data/trailer_dl_info.csv", update_from_files: bool = True
) -> List[TrailerDownloadOptions]:
    """Load trailer download info from a csv file"""
    file = pd.read_csv(path, index_col="index")
    trailer_dl_options: List[TrailerDownloadOptions] = []

    downloaded_audio = [
        int(os.path.splitext(file)[0]) for file in os.listdir(AUDIO_DL_PATH)
    ]
    downloaded_video = [
        int(os.path.splitext(file)[0]) for file in os.listdir(VIDEO_DL_PATH)
    ]
    file.loc[downloaded_audio, "audio_downloaded"] = True
    file.loc[downloaded_video, "video_downloaded"] = True

    for row in file.itertuples():
        trailer_dl_options.append(TrailerDownloadOptions(*row))

    return trailer_dl_options


def download_all(
    selected_opts: List[TrailerDownloadOptions],
    all_opts: List[TrailerDownloadOptions],
    audio_or_video: str,
    redownload=False,
    save_every=25,
):
    """Download all videos in a list"""
    logger = logging.getLogger(__name__)
    if redownload:
        to_download = selected_opts
    else:
        to_download = [
            opt
            for opt in selected_opts
            if opt.download_status(audio_or_video) is None
        ]
    logger.info("Starting download of %s trailers.", len(to_download))
    for idx, dl_option in enumerate(to_download):

        download_yt(
            dl_option,
            audio_or_video=audio_or_video,
        )
        if idx % save_every == 0:
          save_trailer_dl_options(all_opts)
    success = [opt for opt in to_download if opt.download_status(audio_or_video)]
    logger.info(
        "Sucessfull downloads: %s, failed: %s",
        len(success),
        len(to_download) - len(success),
    )


def setup_local_files():
    """Create local folders if they don't exist yet."""

    logger = logging.getLogger(__name__)

    def _create_if_not_exists(path: str):
        if not os.path.exists(path):
            logger.debug("Path %s doesn't exist, creating now.")
            os.mkdir(path)

    paths: List[str] = [DATA_DIR, DOWNLOADS_DIR, AUDIO_DL_PATH, VIDEO_DL_PATH]
    for path in paths:
        _create_if_not_exists(path)


def setup_trailer_dl_options() -> List[TrailerDownloadOptions]:
    """Load trailer download options"""
    logger = logging.getLogger(__name__)
    if not os.path.exists(trailer_info_path):
        logger.debug("Couldn't find path %s, parsing from raw metadata instead.")
        trailer_dl_options = parse_trailer_dl_options()
        save_trailer_dl_options(trailer_dl_options)
    else:
        trailer_dl_options = load_trailer_dl_info()

    return trailer_dl_options


def init_logger(
    log_level: int = logging.INFO,
    log_stdout: bool = True,
    log_file: bool = True,
    log_filename: str = "download.log",
):
    # Logging init
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def download_trailers(
    indices: List[int],
    log_level: int = logging.INFO,
    log_stdout: bool = True,
    log_file: bool = True,
    log_filename: str = "download.log",
):

    logger = init_logger(
        log_level=log_level,
        log_stdout=log_stdout,
        log_file=log_file,
        log_filename=log_filename,
    )

    # Start running
    logger.info("Starting download of trailers.")
    setup_local_files()
    trailer_dl_options = setup_trailer_dl_options()
    if not trailer_dl_options:
        logger.info("Downloaded all files!")
        sys.exit()

    opts_to_download = [trailer_dl_options[i] for i in indices]

    start = time.time()
    # Test downloading first 20
    download_all(opts_to_download,trailer_dl_options, audio_or_video="audio")
    logger.info("Done downloading trailers!")
    save_trailer_dl_options(trailer_dl_options)
    end = time.time()
    logger.info(f"{end - start} seconds elapsed.")
    logger.info("Done")


# Piece of code that can find trailers by name
metadata = load_metadata(path="data/metadata.json")
find = "star wars"
finds = []
for idx, trailer_data in metadata.items():
    if find in trailer_data['imdb']['title'].lower():
      print(trailer_data)
      print(idx)
      finds.append((trailer_data, idx))
print(finds)

# Download the trailers
download_trailers([8001])