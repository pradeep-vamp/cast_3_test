import logging
import multiprocessing
import time
from multiprocessing.dummy import Pool as ThreadPool

import flask
import functions_framework
import pandas as pd

from functions import embed_threading, fetch_from_bigquery

query = """
SELECT
  influencer_id,
  sa.social_id,
  sp.name AS social_platform,
  handle,
  social_token,
  refresh_token,
  DATE(sa.updated_at) AS updated_at
FROM
  `vamp-dw-prod.source_servalan_public.social_accounts` sa
  JOIN `vamp-dw-prod.source_servalan_public.social_platforms` AS sp ON sp.id = sa.social_platform_id
  JOIN `vamp-dw-prod.source_servalan_public.influencers` AS i ON i.id = sa.influencer_id
WHERE
  social_token IS NOT NULL
  AND sa.token_valid = TRUE
  AND followers_count > 5000
  AND DATE(sa.updated_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
ORDER BY 7 DESC
LIMIT 5
;
"""


def trigger_processing():
    logging.info(
        "Data fetched from vamp-dw-prod.source_servalan_public.social_accounts!")
    data = fetch_from_bigquery(query)
    logging.info("Using thread pools to split the task.")
    cpu_count = int(multiprocessing.cpu_count()) - 1
    pool = ThreadPool(cpu_count)
    thread_data = data.to_dict('records')
    t0 = time.time()
    results = pool.map(embed_threading, thread_data)
    results_df = pd.concat(results)
    results_df = results_df.loc[results_df['embeddings'].isna() == False]
    t1 = time.time()

    # write_df_to_bq(table_name="vamp-cast-staging.source_cast_vectorstore.cast_vector_test2", df=data, chunk_size=4000)
    logging.info(
        "Data written in vamp-cast-staging.source_cast_vectorstore.cast_vector!")
    return_msg = data['social_platform'].value_counts()
    data.drop(data.index, inplace=True)
    logging.info(f"Time taken to process: {t1-t0}")
    logging.info(return_msg)
    return return_msg


@functions_framework.http
def run_function(request: flask.Request) -> flask.typing.ResponseReturnValue:
    return trigger_processing()


@functions_framework.http
def hello(request: flask.Request) -> flask.typing.ResponseReturnValue:
    return "Hello world!"
