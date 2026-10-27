import boto3
import time
import os
from model.my_logging import logger

AMI_ID = "ami-00e6d9ae5d82bb643"  # Replace with your AMI ID
INSTANCE_TYPE = "g4dn.xlarge"  # GPU instance type
KEY_NAME = "my-gpu-key"  # SSH
SECURITY_GROUP_ID = "sg-0da6779aa2fe3554f"  # launch-wizard-18
SUBNET_ID = "subnet-06b3090e182a36da4"
IAM_INSTANCE_PROFILE = {"Name": "AuraTrainingInstance"}
TRAINING_JOB_ID = 97


def main(training_job_id, instance_type=INSTANCE_TYPE):
    # Initialize clients
    ec2 = boto3.client(
        "ec2",
        region_name="us-west-1",
        aws_access_key_id=os.environ["S3_AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_AWS_SECRET_ACCESS_KEY"],
    )
    ssm = boto3.client(
        "ssm",
        region_name="us-west-1",
        aws_access_key_id=os.environ["S3_AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_AWS_SECRET_ACCESS_KEY"],
    )

    logger.info("Launching EC2 instance...")
    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SECURITY_GROUP_ID],
        SubnetId=SUBNET_ID,
        IamInstanceProfile=IAM_INSTANCE_PROFILE,
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "AuraTrainingInstance"}]}],
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 50,
                    "VolumeType": "gp2",
                    "DeleteOnTermination": True,
                },
            }
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    logger.info(f"Instance launched: {instance_id}")

    logger.info("Waiting for EC2 instance to be in 'running' state...")
    ec2.get_waiter("instance_status_ok").wait(InstanceIds=[instance_id])

    logger.info("Waiting for SSM agent to register...")

    prepare_command = [
        "export HOME=/root",
        "cd $HOME",
        "/opt/pytorch/bin/pip install python-dotenv",
        "/opt/pytorch/bin/pip install ultralytics",
        "/opt/pytorch/bin/pip install boto3",
        "/opt/pytorch/bin/pip install fire",
        "/opt/pytorch/bin/pip install pymysql",
        "/opt/pytorch/bin/pip install dbutils",
        "/opt/pytorch/bin/pip install pyzipper",
    ]

    write_code_command = [
        "export HOME=/root",
        "cd $HOME",
        "aws s3 cp s3://aura-testing-bucket/auto_train.py auto_train.py",
        "aws s3 cp s3://aura-testing-bucket/env .env",
    ]

    model_training_command = [
        "export HOME=/root",
        "cd $HOME",
        f"/opt/pytorch/bin/python auto_train.py --training_job_id={training_job_id} > training_log.log",
    ]

    all_commands = [prepare_command, write_code_command, model_training_command]

    # 4. Send command via SSM
    for commands in all_commands:
        logger.info("Sending command to instance via SSM...")
        logger.info(f"Command: \n{commands}")
        logger.info("--" * 80)
        command_response = ssm.send_command(InstanceIds=[instance_id], DocumentName="AWS-RunShellScript", Parameters={"commands": commands})

        command_id = command_response["Command"]["CommandId"]
        logger.info(f"Command sent. ID: {command_id}")

        # 5. Wait and get command result
        while True:
            time.sleep(5)  # Give it some time before checking
            output = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)

            status = output["Status"]
            logger.info(f"\033[90mCommand status: {status}\033[0m")
            if status in ["Pending", "InProgress"]:
                logger.info(output["StandardOutputContent"])

            if status in ["Success"]:
                logger.info("\033[90m" + "-" * 100 + "\033[0m")
                logger.info("\033[94m" + output["StandardOutputContent"] + "\033[0m")
                break
            if status in ["Cancelled", "Failed", "TimedOut"]:
                logger.info("\033[90m" + "-" * 100 + "\033[0m")
                logger.info("\033[91m" + output["StandardErrorContent"] + "\033[0m")
                break

    logger.info("Terminating instance...")
    ec2.terminate_instances(InstanceIds=[instance_id])
    logger.info("Termination initiated.")


if __name__ == "__main__":
    training_job_id = 98
    main(training_job_id)
