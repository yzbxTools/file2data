#!/bin/bash

set -ue

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <image_name>:[<image_tag>]"
    echo "Example: $0 hello-world:latest"
    exit 1
fi

img_name=$(echo $1 | cut -d':' -f1)
img_tag=$(echo $1 | cut -d':' -f2)
if [ $img_tag == $1 ]; then
    img_tag=latest
    echo "No img tag specified, using 'latest' as default."
fi

AWS_REGION=us-west-2
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="$img_name"
IMAGE_TAG="$img_tag"
LOCAL_IMAGE="$img_name:$IMAGE_TAG"
echo "Local image: $LOCAL_IMAGE"

# 登录到 ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 081071282306.dkr.ecr.us-west-2.amazonaws.com/cube-studio/yoloworld:h200
# 创建仓库 (如果不存在)
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION || aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION

# 标记镜像
docker tag $LOCAL_IMAGE $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG

# 推送镜像
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG