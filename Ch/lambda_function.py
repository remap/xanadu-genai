import json
import boto3
import logging
import os
from xanadu import Xanadu

logger = logging.getLogger()
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger.setLevel(log_level)

def preprocess_function(xanadu):
  # Eventually preprocess/munge the image, do nothing for now
  logger.info('preprocess function called')
  return xanadu.media, 200

def lambda_handler(event, context):
  error_code = 200 # could remove this if only used once

  # Hardcoded Variables based on namespace
  module = 'ch1'

  buckets = {
    "input_bucket": 'dev-xanadu-raw-input',
    "output_bucket": 'dev-xanadu-preprocess',
    "next_bucket": 'dev-xanadu-inference',
  }

  arns = {
    'job_succeeded_arn': '',
    'job_failed_arn': '',
    'next_topic_arn': 'arn:aws:sns:us-west-2:976618892613:dev-xanadu--ch1--preprocess--to--inference',
  }

  xanadu = Xanadu(logger, event, module, buckets, arns)
  if (xanadu.error_code != 200):
    xanadu.let_firebase_know_job_failed(event, xanadu.error_code)
    return

  logger.info('xanadu initialized')

  preprocessed_media, error_code = preprocess_function(xanadu)
  if (error_code != 200):
    xanadu.let_firebase_know_job_failed(xanadu.metadata, error_code)
    return

  logger.info('xanadu media preprocessed')

  preprocessed_media_arn, error_code = xanadu.put_media(preprocessed_media, xanadu.media_type)
  if (error_code != 200):
    xanadu.let_firebase_know_job_failed(xanadu.metadata, error_code)
    return

  logger.info('xanadu media uploaded')

  next_metadata = xanadu.prep_next_metadata()
  next_metadata_arn, error_code = xanadu.put_metadata(next_metadata)
  if (error_code != 200):
    xanadu.let_firebase_know_job_failed(xanadu.metadata, error_code)
    return

  logger.info('xanadu new metadata uploaded')

  xanadu.let_firebase_know_job_is_done(next_metadata)

  logger.info('xanadu firebase job completion confirmation sent')

  next_message = {
      "default": json.dumps({
          "media_arn": preprocessed_media_arn,
          "metadata_arn": next_metadata_arn
      }),
      "https": json.dumps(next_metadata)
  }

  xanadu.trigger_next_job(next_message)
