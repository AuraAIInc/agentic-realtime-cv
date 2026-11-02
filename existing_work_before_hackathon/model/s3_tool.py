import logging

logger = logging.getLogger(__name__)


def generate_presigned_url(s3_client, bucket_name, object_key, expiration=3600):
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
        return url
    except Exception as e:
        logger.error(f"Error generating pre-signed URL: {e}")
        return None
