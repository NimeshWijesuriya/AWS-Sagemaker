```python

```

# Machine Learning Model with AWS SageMaker

This repository contains a sample code demonstrating how to build, train, and deploy a machine learning model using AWS SageMaker. The example focuses on using XGBoost, a popular gradient boosting algorithm, and involves working with Amazon S3 and Boto3.

## Overview

### AWS SageMaker

[AWS SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that provides tools to build, train, and deploy machine learning models quickly. It abstracts many of the complexities involved in machine learning workflows and integrates seamlessly with other AWS services.

### Amazon S3

[Amazon S3](https://aws.amazon.com/s3/) (Simple Storage Service) is a scalable object storage service that allows you to store and retrieve any amount of data. In this workflow, S3 is used to store datasets and the trained model.

### Boto3

[Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) is the Amazon Web Services (AWS) SDK for Python. It provides an interface to interact with AWS services, including S3 and SageMaker.

## Code Walkthrough

### 1. Importing Libraries

```python
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker import Session
from sagemaker.inputs import TrainingInput


## SageMaker and boto3 Overview

- **SageMaker**: SageMaker's Python SDK provides APIs for building and deploying models.
- **boto3**: AWS SDK for Python, used for interacting with AWS services like S3.
- **get_image_uri**: Helper function to retrieve the image URI for the specified SageMaker algorithm.

## Creating an S3 Bucket and Setting Output Path

```python
prefix = 'xgboost-as-a-built-in-algo'
output_path = 's3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)


### Prefix and Output Path

- **prefix**: A string used to create a unique path within the S3 bucket for the output.
- **output_path**: The S3 path where the trained model will be saved.

### Downloading and Uploading the Dataset

```python
import pandas as pd
import urllib

try:
    urllib.request.urlretrieve("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean-27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ', e)

try:
    model_data = pd.read_csv('./bank_clean.csv', index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ', e)


- **urllib.request.urlretrieve:** Downloads the dataset from a specified URL.
- **pandas:** Used for data manipulation and analysis.
## Code for Data Splitting and Saving to S3t.

### Data Splitting

```python
import numpy as np

# Split the dataset into training and test sets
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                 [int(0.7 * len(model_data))])

print(train_data.shape, test_data.shape)


###  Saving Train And Test Into Buckets

```python
import os
import boto3
import pandas as pd
from sagemaker.inputs import TrainingInput

# Concatenate and save training data
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)],
          axis=1).to_csv('train.csv', index=False, header=False)

# Upload the file to S3
boto3.Session().resource('s3').Bucket(bucket_name).Object(
    os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')

# Define the S3 input for training
s3_input_train = TrainingInput(
    s3_data='s3://{}/{}/train/train.csv'.format(bucket_name, prefix),
    content_type='text/csv'
)

# Concatenate and save testing data
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)],
          axis=1).to_csv('test.csv', index=False, header=False)

# Upload the file to S3
boto3.Session().resource('s3').Bucket(bucket_name).Object(
    os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

# Define the S3 input for testing
s3_input_test = TrainingInput(
    s3_data='s3://{}/{}/test/test.csv'.format(bucket_name, prefix),
    content_type='text/csv'
)


## XGBoost Container Retrieval with SageMaker

This section demonstrates how to automatically retrieve the Amazon SageMaker XGBoost container image URI and build an XGBoost container using the SageMaker Python SDK and `boto3`.

### Code

```python
import boto3
from sagemaker import image_uris

# Define the version of the XGBoost container image
repo_version = '1.0-1'

# Automatically look for the XGBoost image URI and build an XGBoost container
container = image_uris.retrieve('xgboost',
                                boto3.Session().region_name,
                                versin=repo_version)
rsion)

## Conclusion

This repository provides a basic workflow for using AWS SageMaker to build, train, and deploy a machine learning model with XGBoost. For more detailed information on AWS services used in this example, refer to their official documentation:

- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
- [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)



```python

```
