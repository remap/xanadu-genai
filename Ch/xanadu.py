import json
from botocore.utils import ArnParser
import boto3
from urllib.parse import urlparse
import urllib.request
import logging
import os

ERROR_CODE = {
  404: 'Not Found',
  422: 'Unprocessable Entity',
  500: 'Internal Server Error',
}

# Need input and output bucket if we want to go dynamic here
REQUIRED_PARAMS = {
  'media_arn': str,
  'metadata_arn': str
}

REQUIRED_METADATA = {
  'media_file': str,
  'metadata_file': str,
  'input_bucket': str,
  'output_bucket': str,
  'instance': str,
  'target_environment': str,
  'user': str,
  'group': str,
}

LINK_EXPIRES_IN = 3600

ALLOWED_DOWNLOAD_DOMAINS = [
  "https://replicate.delivery",
  "https://www.ucla.edu"
]

class Xanadu:
  media_file_name = None
  media_type = None
  media = None
  metadata_file_name = None
  metadata = None

  def __init__(self, logger, event, module, buckets, arns):
    self.error_code = 200

    self.logger = logger

    self.logger.info('xanadu __init__')

    self.event = event
    self.module = module
    self.input_bucket = buckets['input_bucket']
    self.output_bucket = buckets['output_bucket']
    self.next_bucket = buckets['next_bucket']
    self.job_succeeded_arn = arns['job_succeeded_arn']
    self.job_failed_arn = arns['job_failed_arn']
    self.next_topic_arn = arns['next_topic_arn']
    self.s3 = boto3.client('s3')
    self.arn_parser = ArnParser()

    # Parse required SNS Message json
    try:
      message = event.get('Records', [])[0].get('Sns', {}).get('Message', '{}')
      self.data = json.loads(message)
    except Exception as e:
      self.error_code = 400
      self.logger.error({f"Unable to parse input message: {event}\nBaseException caught: {type(e).__name__} - {e}"})
      return

    self.logger.info('xanadu message parsed')

    if not self.validate_required_input():
      self.error_code = 400
      self.logger.error({f"Unable to validate input: {self.data}"})
      return

    self.logger.info('xanadu required variables set')

    self.error_code = self.set_media()
    if (self.error_code != 200):
      self.logger.error({f"Unable to instantiate media object"})
      return

    self.logger.info('xanadu media object set')

    self.error_code = self.set_metadata()
    if (self.error_code != 200):
      self.logger.error({f"Unable to instantiate metadata object"})
      return

    self.logger.info('xanadu metadata object set')

    return

  def validate_data_using_definition(self, definition, data):
    self.logger.info('xanadu validate_data_using_definition')
    for param, expected_type in definition.items():
      if param not in data:
        self.logger.debug(f"{param} is not set in message")
        return False
      if not isinstance(data[param], expected_type):
        self.logger.debug(f"{param} is not the expected type {expected_type}")
        return False

    return True

  def validate_required_input(self):
    self.logger.info('xanadu validate_required_input')
    return self.validate_data_using_definition(REQUIRED_PARAMS, self.data)

  def validate_metadata(self, metadata):
    self.logger.info('xanadu validate_metadata')
    return self.validate_data_using_definition(REQUIRED_METADATA, metadata)

  def get_s3_object_arn_parts(self, expected_bucket, arn):
    log_message = {
      "function": "get_s3_object_arn_parts",
      "expected_bucket": expected_bucket,
      "arn": arn
    }
    self.logger.debug(repr(log_message))

    error_code = 200
    parsed_arn = None

    try:
      arn_parser = ArnParser()
      arn_parts = arn_parser.parse_arn(arn)

      if arn_parts['service'] != 's3':
        return None, 400

      resource = arn_parts['resource']
      bucket, key = resource.split('/', 1)

      if bucket != expected_bucket:
        self.logger.error('Invalid bucket supplied')
        return None, 400

      parsed_arn = {
        "bucket": bucket,
        "key": key
      }
    except Exception as e:
      self.logger.error(f"Error parsing s3 arn\nBaseException caught: {type(e).__name__} - {e}")
      # Could return the InvalidArnException?
      return None, 400

    return parsed_arn, error_code

  def get_s3_uri_parts(self, expected_bucket, s3_uri):
    log_message = {
      "function": "get_s3_uri_parts",
      "expected_bucket": expected_bucket,
      "s3_uri": s3_uri
    }
    self.logger.debug(repr(log_message))

    try:
      parsed_url = urlparse(s3_uri)
      if parsed_url.scheme != 's3':
        self.logger.debug(f"Invalid S3 URI: {s3_uri}")
        return None, 422
    except Exception as e:
      self.logger.error(f"Error parsing s3 uri\nBaseException caught: {type(e).__name__} - {e}")
      return None, 422

    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')

    s3_uri_parts = {
      "bucket": bucket,
      "key": key
    }

    return s3_uri_parts, 200

  def is_url(self, url):
    try:
      result = urlparse(url)
      return all([result.scheme, result.netloc])
    except AttributeError:
      return False

  def create_anonymous_url_to_s3_obj(self, bucket, key):
    error_code = 200
    url = None

    try:
      url = self.s3.generate_presigned_url(
        'get_object',
        Params={
          'Bucket': bucket,
          'Key': key
        },
        ExpiresIn=LINK_EXPIRES_IN
      )

      if not self.is_url(url):
        self.logger.error(f"Error: invalid presigned url from s3")
        return None, 500

      return url, error_code

    except Exception as e:
      self.logger.error(f"Error getting s3 obj url\nBaseException caught: {type(e).__name__} - {e}")
      return None, 500

  def set_media(self):
    self.logger.info('xanadu set_media')
    s3_parts, error_code = self.get_s3_object_arn_parts(
      self.input_bucket, self.data['media_arn']
    )

    if (error_code != 200):
      return error_code

    response, error_code = self.get_object_from_s3(s3_parts['bucket'], s3_parts['key'])

    if (error_code != 200):
      return error_code

    content = response['Body'].read()

    if not content:
      return None, 400

    anon_url, error_code = self.create_anonymous_url_to_s3_obj(s3_parts['bucket'], s3_parts['key'])

    if (error_code != 200):
      return error_code

    if not anon_url:
      return None, 400

    self.media_file_name = os.path.basename(s3_parts['key'])
    self.media_type = response['ContentType']
    self.media = content
    self.media_url = anon_url

    return error_code

  def set_metadata(self):
    self.logger.info('xanadu set_metadata')
    s3_parts, error_code = self.get_s3_object_arn_parts(
      self.input_bucket, self.data['metadata_arn']
    )

    if (error_code != 200):
      return error_code

    response, error_code = self.get_object_from_s3(s3_parts['bucket'], s3_parts['key'])
    if (error_code != 200):
      return error_code

    content = response['Body'].read()

    if not content:
      return None, 400

    try:
      data = json.loads(content)
      if not self.validate_metadata(data):
        return 400

      self.metadata_file_name = os.path.basename(s3_parts['key'])
      self.metadata = data
      return error_code
    except Exception as e:
      self.logger.error(f"Error loading metadata\nBaseException caught: {type(e).__name__} - {e}")
      return 422

  def get_file_path(self):
    return f"{self.metadata['instance']}/{self.module}/{self.metadata['user']}/upload/"

  def get_object_from_s3(self, bucket, key):
    error_code = 200

    try:
      response = self.s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
      self.logger.error(f"Error getting object from s3\nBaseException caught: {type(e).__name__} - {e}")
      return None, 400

    return response, error_code

  def put_media(self, content, type):
    error_code = 200
    key = self.get_file_path() + self.media_file_name
    self.logger.info(f"xanadu put_media, 's3://{self.output_bucket}/{key}'")
    new_arn, error_code = self.put_file_contents(self.output_bucket, key, content, type)
    if (error_code != 200):
      self.logger.error(f"xanadu Unable to put_media: 's3://{self.output_bucket}/{key}'")
      return error_code

    self.logger.debug(f"xanadu put_media new arn 's3://{self.output_bucket}/{key}'")

    return new_arn, error_code

  def put_metadata(self, content):
    error_code = 200
    key = self.get_file_path() + self.metadata_file_name
    self.logger.info(f"xanadu put_metadata, 's3://{self.output_bucket}/{key}'")
    new_arn, error_code = self.put_file_contents(self.output_bucket, key, content, 'application/json')
    if (error_code != 200):
      self.logger.error(f"xanadu Unable to put_metadata: 's3://{self.output_bucket}/{key}'")
      return None, error_code

    return new_arn, error_code

  def put_file_contents(self, bucket, key, content, content_type):
    error_code = 200

    try:
      self.s3.put_object(
          Bucket=bucket,
          Key=key,
          Body=content,
          ContentType=content_type
      )
    except Exception as e:
      self.logger.error(f"Error putting object to s3\nBaseException caught: {type(e).__name__} - {e}")
      return None, 500

    new_arn = f"arn:aws:s3:::{bucket}/{key}"

    return new_arn, error_code

  def prep_next_metadata(self):
    log_message = {
      "function": "prep_next_metadata",
      "metadata": self.metadata,
    }
    self.logger.debug(repr(log_message))

    next_metadata = self.metadata
    next_metadata['input_bucket'] = self.output_bucket
    next_metadata['output_bucket'] = self.next_bucket

    return json.dumps(next_metadata)

  def get_prompt_from_firebase(self, data):
    self.logger.info('xanadu get_prompt_from_firebase')
    error_code = 200

    instance = self.metadata.instance
    group = self.metadata.group

    prompt = f"""
      This is a placeholder prompt, replace with call to firebase using:
        instance: {instance}
        group: {group}
    """

    return prompt, error_code

  def is_allowed_domain(self, url):
    try:
      parsed = urlparse(url)
      if parsed.scheme  != "https":
        return False

      return parsed.netloc in ALLOWED_DOWNLOAD_DOMAINS
    except Exception:
      return False

  def load_file_from_url(self, url):
    if not self.is_allowed_domain(url):
      self.logger.error(f"Error: cannot load file from invalid url")
      return None, 500

    try:
      response = urllib.request.urlopen(url)
      return response, 200

    except Exception as e:
      self.logger.error(f"Error loading file from url\nBaseException caught: {type(e).__name__} - {e}")
      return None, 500

  def get_error_message(self, error_code):
    if error_code in ERROR_CODE:
      return ERROR_CODE[error_code]
    else:
      return 'undefined error'


  def publish_sns(self, topic_arn, message):
    self.logger.info('xanadu publish_sns')

    sns_client = boto3.client('sns')
    error_code = 200

    try:
      topic_arn_parts = self.arn_parser.parse_arn(topic_arn)

      if topic_arn_parts['service'] != 'sns':
        self.logger.info(f"topic_arn is not an sns service arn")
        return 400

    except Exception as e:
      self.logger.error(f"Error parsing topic arn\nBaseException caught: {type(e).__name__} - {e}")
      return 500

    try:
      string_message = json.dumps(message)
    except Exception as e:
      self.logger.error(f"Error parsing message\nBaseException caught: {type(e).__name__} - {e}")
      return 500

    try:
      self.logger.info(f"String Message: {string_message}")

      response = sns_client.publish(
        TopicArn=topic_arn,
        Message=string_message,
        MessageStructure='json'
      )

      log_message = {
        "function": "publish_sns",
        "message": "Message published successfully.",
        "sns_response": response
      }
      self.logger.info(repr(log_message))
    except Exception as e:
      self.logger.error(f"Error publishing message\nBaseException caught: {type(e).__name__} - {e}")
      error_code = 500

    return error_code

  def let_firebase_know_job_failed(self, message, error_code):
    # trigger sns message to notify firebase?
    log_message = {
        "function": "let_firebase_know_job_failed",
        "job_failed_arn": self.job_failed_arn,
        "message": message,
        "error_code": error_code,
        "error_message": self.get_error_message(error_code),
    }
    self.logger.info(repr(log_message))

    firebase_message = {
        "message": message,
        "error_code": error_code,
        "error_message": self.get_error_message(error_code),
    }

    if bool(self.job_failed_arn):
      return self.publish_sns(self.job_failed_arn, firebase_message)
    else:
      self.logger.error(f"No job_failed_arn to run")
      return 400

  def let_firebase_know_job_is_done(self, message):
    # trigger sns message to notify firebase?
    log_message = {
        "function": "let_firebase_know_job_is_done",
        "job_succeeded_arn": self.job_succeeded_arn,
        "message": message,
    }
    self.logger.info(repr(log_message))

    if bool(self.job_succeeded_arn):
      return self.publish_sns(self.job_succeeded_arn, message)
    else:
      self.logger.error(f"No job_succeeded_arn to run")
      return 400

  def trigger_next_job(self, message):
    # trigger sns message to notify firebase?
    log_message = {
        "function": "trigger_next_job",
        "next_topic_arn": self.next_topic_arn,
        "message": message,
    }
    self.logger.info(repr(log_message))

    if bool(self.next_topic_arn):
      return self.publish_sns(self.next_topic_arn, message)
    else:
      self.logger.error(f"No next_topic_arn to run")
      return 400
