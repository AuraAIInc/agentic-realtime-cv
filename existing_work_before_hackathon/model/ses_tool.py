from dotenv import load_dotenv

load_dotenv()
import os
import boto3
from botocore.exceptions import ClientError
from model.my_logging import logger


def send_email(email: str, subject: str, context: str) -> bool:
    logger.info(context)
    client = boto3.client("ses", 
                          region_name=os.getenv("SES_AWS_DEFAULT_REGION"),
                          aws_access_key_id=os.getenv("SES_AWS_ACCESS_KEY"), 
                          aws_secret_access_key=os.getenv("SES_AWS_SECRET_ACCESS_KEY"))
    try:
        response = client.send_email(
            Destination={
                "ToAddresses": [email],
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": "UTF-8",
                        "Data": context,
                    },
                    "Text": {
                        "Charset": "UTF-8",
                        "Data": context.replace("<br>", "\n"),  # Plain text fallback
                    },
                },
                "Subject": {
                    "Charset": "UTF-8",
                    "Data": subject,
                },
            },
            Source=os.getenv("SES_FROM_EMAIL"),
        )
        return True
    except ClientError as e:
        print(f"Error sending email: {e.response['Error']['Message']}")
        return False


if __name__ == "__main__":
    send_email(os.getenv("SES_FROM_EMAIL"), "Test Subject", "Test Context")
