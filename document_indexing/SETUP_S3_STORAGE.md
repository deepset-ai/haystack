## how to setup access to the S3 bucket

1. download the aws cli: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2. type ` aws configure --profile fsdl22`, fill in the acces key and id that you have received. Details: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
3. to see if you succeeded, in your terminal type `aws s3 ls s3://board-games-rules-explainer/`
