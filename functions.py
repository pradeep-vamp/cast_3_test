import json
import logging
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import vertexai
from dotenv import load_dotenv
from google.cloud import bigquery
from more_itertools import sliced
from vertexai.vision_models import Image, MultiModalEmbeddingModel

load_dotenv()

tiktok_access_token = os.environ['TIKTOK_TCM_ACCESS_TOKEN']
tcm_account_id = os.environ['TIKTOK_TCM_ACCOUNT_ID']

client_id = os.environ['TIKTOK_CLIENT_ID']
client_secret = os.environ['TIKTOK_CLIENT_SECRET']

vertexai.init(location="europe-west1")
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")


def write_df_to_bq(table_name=None, df=pd.DataFrame, chunk_size=None):
    """
    Executes sql query provided

    client : bigquery.Client() connection
    table_name : "your-project.your_dataset.your_table_name"
    chunk_size : Integer
    df : sql query as string format

    returns : results boolean
    """
    logger = logging.getLogger(__name__)

    # Construct a BigQuery client object.
    client = bigquery.Client()

    if table_name is None:
        logging.exception("Missing table_name values")
        return False

    if df is None:
        logging.exception("Missing dataframe")
        return False

    if chunk_size is None:
        chunk_size = 5000

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"
    )

    total_rows_count = len(df)
    loaded_rows = 0

    for index_slice in sliced(range(len(df)), chunk_size):
        small_chunk_df = df.iloc[index_slice]
        cur_chunk_size = len(small_chunk_df)
        logger.info("Now loading {} out of {} rows".format(
            cur_chunk_size, (total_rows_count - loaded_rows)))

        job = client.load_table_from_dataframe(
            small_chunk_df, table_name, job_config=job_config)
        job.result()

        loaded_rows += cur_chunk_size
        logger.info(
            "Loaded {} rows into bq {} with remaining {} rows to be loaded".format(
                job.output_rows, table_name, (total_rows_count - loaded_rows)
            )
        )

    return True


def fetch_from_bigquery(query_string, query_parameters=None) -> pd.DataFrame:
    """
    Fetch data from BigQuery and return as a pandas DataFrame
    """
    logging.info("Fetching data from BigQuery")
    bqclient = bigquery.Client(location="europe-west1")
    job_config = None

    if query_parameters is not None:
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)

    dataframe = (
        bqclient.query(query_string, job_config=job_config)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )
    return dataframe


def read_image(url):
    """
    Read image from URL
    """
    logging.info(f"Reading image from URL: {url}")
    response = requests.get(url)
    img = BytesIO(response.content)
    return img


def retrieve_tiktok_data(handle, client_id=client_id, client_secret=client_secret, access_token=tiktok_access_token, tcm_account_id=tcm_account_id):
    """
    Retrieve TikTok data for a given handle
    """
    logging.info(f"Retrieving TikTok data for handle: {handle}")
    url = 'https://business-api.tiktok.com/open_api/v1.3/tcm/creator/public/video/list/'
    response = requests.get(url, params={'handle_name': handle, 'tcm_account_id': tcm_account_id}, headers={
                            'Access-Token': access_token})
    raw = json.loads(response.text)
    if raw['code'] == 0:
        result = pd.DataFrame(raw['data']['posts'])
    else:
        result = pd.DataFrame(columns=['display_name', 'comments', 'embed_url', 'video_id',
                              'shares', 'thumbnail', 'likes', 'create_time', 'video_views', 'caption'])

    result.rename(columns={'thumbnail_url': 'thumbnail'}, inplace=True)
    return result


def call_tiktok_discovery(params, access_token=tiktok_access_token):
    url = 'https://business-api.tiktok.com/open_api/v1.3/tcm/creator/discover/'
    response = requests.post(url, params=params, headers={
                             'Access-Token': access_token, 'Content-Type': 'application/json'})
    raw = json.loads(response.text)

    return raw


def retrieve_meta_media(account_id, access_token):
    """
    Retrieve meta media for a given account_id
    """
    url = "https://graph.facebook.com/v12.0/{account_id}?fields=".format(
        account_id=account_id)+"biography, media{media_url,media_type, image_url, cover_url, thumbnail_url,caption}"+f"&access_token={access_token}"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    vals = json.loads(response.text)
    video_list = []
    for i in vals['media']['data']:
        video_list.append({'caption': i.get('caption'), 'thumbnail': i.get(
            'thumbnail_url', i.get('media_url')), 'media_url': i.get('media_url')})
    return pd.DataFrame(video_list)


def retrieve_youtube_uploads(social_id):
    """
    Retrieve Youtube uploads for a given social_id
    """
    logging.info(
        f"Retrieving Youtube uploads for social_id: {social_id}")
    url = 'https://youtube.googleapis.com/youtube/v3/channels'
    response = requests.get(url, params={
                            'part': 'ContentDetails', 'key': os.environ['YOUTUBE_DATA_API_KEY'], 'id': social_id})
    raw = json.loads(response.text)
    return raw['items'][0]['contentDetails']['relatedPlaylists']['uploads']


def retrieve_video_lists(social_id):
    """
    Retrieve video lists for a given social_id
    """
    uploads = retrieve_youtube_uploads(social_id)
    url = 'https://youtube.googleapis.com/youtube/v3/playlistItems'
    response = requests.get(url, params={
                            'part': 'ContentDetails', 'key': os.environ['YOUTUBE_DATA_API_KEY'], 'playlistId': uploads, 'maxResults': 30})
    raw = json.loads(response.text)
    videoIds = []
    for i in raw['items']:
        videoIds.append(i['contentDetails']['videoId'])
    return videoIds


def retrieve_video_data(videoIds):
    """
    Retrieve video data for a given videoId
    """
    logging.info(f"Retrieving video data for videoIds: {videoIds}")
    url = 'https://youtube.googleapis.com/youtube/v3/videos'
    response = requests.get(url, params={
                            'part': 'snippet', 'key': os.environ['YOUTUBE_DATA_API_KEY'], 'id': ",".join(videoIds), 'maxResults': 30})
    raw = json.loads(response.text)
    videoData = []
    for i in raw['items']:
        videoData.append({"title": i['snippet']['title'], "description":  i['snippet']
                         ['description'], "thumbnail": i['snippet']['thumbnails']['standard']['url']})
    return pd.DataFrame(videoData)


def retrieve_youtube_videos(social_id):
    """
    Retrieve Youtube videos for a given social_id
    """
    logging.info(
        f"Retrieving Youtube videos for social_id: {social_id}")
    video_ids = retrieve_video_lists(social_id)
    data = retrieve_video_data(video_ids)
    data['caption'] = 'Title: ' + data['title'] + \
        ' Description: ' + data['description']
    return data


def retrieve_post_information(social_platform, handle=None, social_id=None, social_token=None):
    """
    Retrieve post information for a given social platform
    """
    logging.info(
        f"Retrieving post information for social platform: {social_platform}")
    try:
        if social_platform == 'Instagram':
            data = retrieve_meta_media(social_id, social_token)
        elif social_platform == 'Youtube':
            data = retrieve_youtube_videos(social_id)
        elif social_platform == 'TikTok':
            data = retrieve_tiktok_data(handle)
        else:
            data = pd.DataFrame(columns=['caption', 'thumbnail'])
        data = data.loc[:, ['caption', 'thumbnail']]
    except Exception as e:
        data = pd.DataFrame(columns=['caption', 'thumbnail'])
        logging.error(e)
    data['caption'] = data['caption'].str.slice(0, 1000)
    return data


def generate_embeddings(thumbnail_url, caption, model=MultiModalEmbeddingModel.from_pretrained("multimodalembedding")):
    """
    Generate embeddings for a given image and caption
    """
    embeddings = model.get_embeddings(image=Image(
        read_image(thumbnail_url).read()), contextual_text=caption)
    result = {'image_embedding': embeddings.image_embedding,
              'caption_embedding': embeddings.text_embedding}
    return result


def retrieve_and_embed(social_platform, handle=None, social_id=None, social_token=None):
    """
    Retrieve and embed data for a given social platform
    """
    logging.info(
        f"Retrieving and embedding data for social platform: {social_platform}")
    try:
        post_info = retrieve_post_information(
            social_platform, handle, social_id, social_token)
        post_info["embeddings"] = post_info.apply(
            lambda x: generate_embeddings(x.thumbnail, x.caption), axis=1)
        post_info["social_platform"] = social_platform
        post_info["handle"] = handle
        post_info["social_id"] = social_id
    except Exception as e:
        logging.error(e)
        post_info = pd.DataFrame({'caption': None, 'thumbnail': None, 'embeddings': None,
                                 'social_platform': social_platform, 'handle': handle, 'social_id': social_id}, index=[0])

    post_info['image_vector'] = pd.Series(
        [x['image_embedding'] for x in post_info['embeddings']])
    post_info['caption_vector'] = pd.Series(
        [x['caption_embedding'] for x in post_info['embeddings']])

    post_info = post_info.loc[post_info['embeddings'].isna() == False]
    creators = post_info.loc[:, [
        'social_platform', 'handle']].drop_duplicates()
    creators['creator_vector'] = pd.Series()
    post_info['media_vector'] = post_info['image_vector'] + \
        post_info['caption_vector']

    for index, row in creators.iterrows():
        social_platform = row['social_platform']
        handle = row['handle']
        try:
            subset = post_info.loc[(post_info['handle'] == handle) & (
                post_info['social_platform'] == social_platform)]
            media_vectors = np.stack(subset['media_vector'])

            creator_vector = np.mean(media_vectors, axis=0)

            post_info['creator_vector'][index] = creator_vector.tolist()

        except Exception as e:
            logging.info(creators)
            logging.error(e)

    creators = creators.loc[creators['creator_vector'].isna() == False]
    post_info = post_info.drop(
        columns=['image_vector', 'caption_vector', 'embeddings'])
    logging.info(post_info.head())
    # Write to BigQuery, to ensure processed data is not lost
    write_df_to_bq(table_name="vamp-cast-staging.source_cast_vectorstore.cast_vector_test3",
                   df=post_info, chunk_size=4000)
    write_df_to_bq(table_name="vamp-cast-staging.source_cast_vectorstore.cast_creators_vector_test1",
                   df=creators, chunk_size=4000)
    return post_info


def embed_threading(item):
    """
    Embed data for a given item
    """
    logging.info(f"Embedding data for item: {item}")
    social_platform = item.get('social_platform')
    handle = item.get('handle')
    social_id = item.get('social_id')
    social_token = item.get('social_token')
    return retrieve_and_embed(social_platform, handle, social_id, social_token)
