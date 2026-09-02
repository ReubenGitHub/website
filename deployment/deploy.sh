#!/bin/bash
# MyWebsite Production Deployment
# Complete deployment: setup AWS resources, build images, push to ECR, deploy to ECS
#
# Usage:
#   ./deploy.sh              - Full deployment (setup + build + deploy)
#   ./deploy.sh setup        - Setup AWS resources only (ECR, ECS, IAM, ALB, ACM)
#   ./deploy.sh build        - Build and push Docker images only
#   ./deploy.sh deploy       - Deploy to ECS only (register task def + update service)
#
# After deployment, follow the Cloudflare setup instructions below.

set -e

# Disable AWS CLI pager to output everything at once
export AWS_PAGER=""

AWS_PROFILE="mywebsite-production"
AWS_REGION="eu-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query 'Account' --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/reubenhow"
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)
CLUSTER="mywebsite-production"
SERVICE="mywebsite-production"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

###############################################################################
# PREREQUISITES
###############################################################################
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=()
    command -v docker &>/dev/null || missing+=("docker")
    command -v aws &>/dev/null    || missing+=("aws-cli")
    command -v git &>/dev/null    || missing+=("git")
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing: ${missing[*]}"
        exit 1
    fi
    
    # Check AWS auth
    if ! aws sts get-caller-identity --profile $AWS_PROFILE &>/dev/null; then
        log_error "AWS not authenticated. Run: aws sso login --profile $AWS_PROFILE"
        exit 1
    fi
    
    log_info "All prerequisites met!"
}

###############################################################################
# SETUP AWS RESOURCES
###############################################################################
setup_ecr() {
    log_info "Setting up ECR repositories..."
    for repo in frontend backend dotnet-api; do
        aws ecr create-repository --profile $AWS_PROFILE --repository-name "reubenhow/$repo" 2>/dev/null || \
            log_warn "reubenhow/$repo already exists"
    done
    log_info "ECR repositories ready!"
}

setup_ecs() {
    log_info "Setting up ECS cluster and IAM roles..."
    
    # Create cluster
    aws ecs describe-clusters --profile $AWS_PROFILE --clusters $CLUSTER &>/dev/null || \
        aws ecs create-cluster --profile $AWS_PROFILE --cluster-name $CLUSTER
    
    # Create execution role
    local role_name="ecsTaskExecutionRole"
    if ! aws iam get-role --role-name $role_name --profile $AWS_PROFILE &>/dev/null; then
        log_info "Creating $role_name..."
        aws iam create-role \
            --role-name $role_name \
            --assume-role-policy-document '{
                "Version":"2012-10-17",
                "Statement":[{
                    "Effect":"Allow",
                    "Principal":{"Service":"ecs-tasks.amazonaws.com"},
                    "Action":"sts:AssumeRole"
                }]
            }' --profile $AWS_PROFILE
        
        aws iam attach-role-policy \
            --role-name $role_name \
            --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
            --profile $AWS_PROFILE
    else
        log_warn "$role_name already exists"
    fi
    
    # Create task role
    local task_role_name="ecsTaskRole"
    if ! aws iam get-role --role-name $task_role_name --profile $AWS_PROFILE &>/dev/null; then
        log_info "Creating $task_role_name..."
        aws iam create-role \
            --role-name $task_role_name \
            --assume-role-policy-document '{
                "Version":"2012-10-17",
                "Statement":[{
                    "Effect":"Allow",
                    "Principal":{"Service":"ecs-tasks.amazonaws.com"},
                    "Action":"sts:AssumeRole"
                }]
            }' --profile $AWS_PROFILE
        
        aws iam attach-role-policy \
            --role-name $task_role_name \
            --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
            --profile $AWS_PROFILE
    else
        log_warn "$task_role_name already exists"
    fi
    
    # Create CloudWatch log groups
    for log_group in /mywebsite/frontend /mywebsite/backend /mywebsite/dotnet-api; do
        if aws logs describe-log-groups --profile $AWS_PROFILE --log-group-name-prefix "${log_group}" --output json 2>/dev/null | python3 -c "import json,sys; data=json.load(sys.stdin); print('yes' if len(data.get('logGroups',[])) > 0 else 'no')" 2>/dev/null | grep -q "yes"; then
            log_warn "Log group $log_group already exists"
        else
            aws logs create-log-group --profile $AWS_PROFILE --log-group-name $log_group &>/dev/null
            log_info "Created log group $log_group"
        fi
    done
    
    log_info "ECS cluster and IAM roles ready!"
}

setup_alb() {
    log_info "Setting up Application Load Balancer..."
    
    # Get default VPC and subnets
    local vpc_id
    vpc_id=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
    local subnets
    subnets=$(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --filters "Name=vpc-id,Values=$vpc_id" --query 'Subnets[].SubnetId' --output text)
    
    # Create ALB security group
    local alb_sg_name="mywebsite-alb-sg"
    local alb_sg_id=""
    
    alb_sg_id=$(aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION \
        --filters "Name=group-name,Values=$alb_sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$alb_sg_id" ] || [ "$alb_sg_id" = "None" ]; then
        log_info "Creating ALB security group..."
        alb_sg_id=$(aws ec2 create-security-group \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-name $alb_sg_name \
            --description "ALB for mywebsite" \
            --vpc-id $vpc_id \
            --output text --query 'GroupId')
        
        # Allow HTTP/HTTPS inbound
        aws ec2 authorize-security-group-ingress \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-id $alb_sg_id \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 2>/dev/null || true
        
        aws ec2 authorize-security-group-ingress \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-id $alb_sg_id \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0 2>/dev/null || true
    else
        log_warn "ALB security group already exists"
    fi
    
    # Create ALB
    local alb_name="mywebsite-alb"
    local alb_arn=""
    
    if ! aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION \
        --names $alb_name &>/dev/null; then
        
        log_info "Creating ALB..."
        alb_arn=$(aws elbv2 create-load-balancer \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --name $alb_name \
            --subnets $subnets \
            --security-groups $alb_sg_id \
            --scheme internet-facing \
            --type application \
            --output text --query 'LoadBalancers[0].LoadBalancerArn')
    else
        alb_arn=$(aws elbv2 describe-load-balancers \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --names $alb_name \
            --output text --query 'LoadBalancers[0].LoadBalancerArn')
        log_warn "ALB already exists"
    fi
    
    # Create target group
    local tg_name="mywebsite-frontend-tg"
    local tg_arn=""
    
    if ! aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION \
        --names $tg_name &>/dev/null; then
        
        log_info "Creating target group..."
        tg_arn=$(aws elbv2 create-target-group \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --name $tg_name \
            --protocol HTTP \
            --port 80 \
            --target-type ip \
            --vpc-id $vpc_id \
            --health-check-path / \
            --health-check-protocol HTTP \
            --health-check-interval-seconds 30 \
            --health-check-timeout-seconds 10 \
            --healthy-threshold-count 3 \
            --unhealthy-threshold-count 3 \
            --output text --query 'TargetGroups[0].TargetGroupArn')
    else
        tg_arn=$(aws elbv2 describe-target-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --names $tg_name \
            --output text --query 'TargetGroups[0].TargetGroupArn')
        log_warn "Target group already exists"
    fi
    
    # Create HTTP listener
    local http_listener_arn=""
    if ! aws elbv2 describe-listeners --profile $AWS_PROFILE --region $AWS_REGION \
        --load-balancer-arn $alb_arn --ports 80 &>/dev/null; then
        
        log_info "Creating HTTP listener (port 80)..."
        http_listener_arn=$(aws elbv2 create-listener \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --protocol HTTP \
            --port 80 \
            --default-action TargetGroupArn=$tg_arn,Type=forward \
            --output text --query 'Listeners[0].ListenerArn')
    else
        http_listener_arn=$(aws elbv2 describe-listeners \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --output text --query 'Listeners[?Port==`80`].ListenerArn')
        log_warn "HTTP listener already exists"
    fi
    
    # Create ACM certificate
    local cert_arn="arn:aws:acm:$AWS_REGION:$ACCOUNT_ID:certificate/3ed4d5a9-167a-47de-ae94-c9a14d83a3b8"
    log_info "Checking ACM certificate..."
    if ! aws acm describe-certificate \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --certificate-arn "$cert_arn" &>/dev/null; then
        
        log_info "Requesting ACM certificate..."
        cert_arn=$(aws acm request-certificate \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --domain-name reubenhow.com \
            --domain-name www.reubenhow.com \
            --validation-method DNS \
            --output text --query 'CertificateArn')
        
        log_warn "Certificate requested - MUST validate in Cloudflare (see instructions below)"
    else
        log_warn "Certificate already exists"
    fi
    
    # Create HTTPS listener
    local https_listener_arn=""
    if ! aws elbv2 describe-listeners --profile $AWS_PROFILE --region $AWS_REGION \
        --load-balancer-arn $alb_arn --ports 443 &>/dev/null; then
        
        log_info "Creating HTTPS listener (port 443)..."
        https_listener_arn=$(aws elbv2 create-listener \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --protocol HTTPS \
            --port 443 \
            --certificates CertificateArn=$cert_arn \
            --default-action TargetGroupArn=$tg_arn,Type=forward \
            --ssl-policy ELBSecurityPolicy-2016-08 \
            --output text --query 'Listeners[0].ListenerArn')
    else
        https_listener_arn=$(aws elbv2 describe-listeners \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --output text --query 'Listeners[?Port==`443`].ListenerArn')
        log_warn "HTTPS listener already exists"
    fi
    
    # Get ALB DNS
    local alb_dns
    alb_dns=$(aws elbv2 describe-load-balancers \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --names $alb_name \
        --output text --query 'LoadBalancers[0].DNSName')
    
    log_info "ALB ready!"
    log_info "ALB DNS: $alb_dns"
    log_info "Target Group: $tg_arn"
    
    # Store ALB DNS for later use
    echo "$alb_dns" > /tmp/alb-dns.txt
}

setup_all() {
    log_info "=== Setting up AWS resources ==="
    local created=0
    local exists=0
    
    # ECR
    log_info "Checking ECR repositories..."
    for repo in frontend backend dotnet-api; do
        if aws ecr describe-repositories --profile $AWS_PROFILE --repository-names "reubenhow/$repo" &>/dev/null; then
            log_warn "  reubenhow/$repo already exists"
            ((exists++)) || true
        else
            aws ecr create-repository --profile $AWS_PROFILE --repository-name "reubenhow/$repo" &>/dev/null
            log_info "  Created reubenhow/$repo"
            ((created++)) || true
        fi
    done
    
    # ECS Cluster
    log_info "Checking ECS cluster..."
    if aws ecs describe-clusters --profile $AWS_PROFILE --clusters $CLUSTER &>/dev/null; then
        log_warn "  Cluster $CLUSTER already exists"
        ((exists++)) || true
    else
        aws ecs create-cluster --profile $AWS_PROFILE --cluster-name $CLUSTER &>/dev/null
        log_info "  Created cluster $CLUSTER"
        ((created++)) || true
    fi
    
    # ECS IAM Roles
    log_info "Checking IAM roles..."
    for role_name in ecsTaskExecutionRole ecsTaskRole; do
        if aws iam get-role --role-name $role_name --profile $AWS_PROFILE &>/dev/null; then
            log_warn "  Role $role_name already exists"
            ((exists++)) || true
        else
            log_info "  Creating $role_name..."
            aws iam create-role \
                --role-name $role_name \
                --assume-role-policy-document '{
                    "Version":"2012-10-17",
                    "Statement":[{
                        "Effect":"Allow",
                        "Principal":{"Service":"ecs-tasks.amazonaws.com"},
                        "Action":"sts:AssumeRole"
                    }]
                }' --profile $AWS_PROFILE &>/dev/null
            
            if [ "$role_name" = "ecsTaskExecutionRole" ]; then
                aws iam attach-role-policy \
                    --role-name $role_name \
                    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
                    --profile $AWS_PROFILE &>/dev/null
            elif [ "$role_name" = "ecsTaskRole" ]; then
                aws iam attach-role-policy \
                    --role-name $role_name \
                    --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
                    --profile $AWS_PROFILE &>/dev/null
            fi
            log_info "  Created $role_name"
            ((created++)) || true
        fi
    done
    
    # CloudWatch Log Groups
    log_info "Checking CloudWatch log groups..."
    for log_group in /mywebsite/frontend /mywebsite/backend /mywebsite/dotnet-api; do
        if aws logs describe-log-groups --profile $AWS_PROFILE --log-group-name-prefix "${log_group}" --output json 2>/dev/null | python3 -c "import json,sys; data=json.load(sys.stdin); print('yes' if len(data.get('logGroups',[])) > 0 else 'no')" 2>/dev/null | grep -q "yes"; then
            log_warn "  Log group $log_group already exists"
            ((exists++)) || true
        else
            aws logs create-log-group --profile $AWS_PROFILE --log-group-name $log_group &>/dev/null
            log_info "  Created $log_group"
            ((created++)) || true
        fi
    done
    
    # ALB Security Group
    log_info "Checking ALB security group..."
    local vpc_id
    vpc_id=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
    local alb_sg_name="mywebsite-alb-sg"
    local alb_sg_id=""
    
    alb_sg_id=$(aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION \
        --filters "Name=group-name,Values=$alb_sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$alb_sg_id" ] || [ "$alb_sg_id" = "None" ]; then
        log_info "  Creating security group $alb_sg_name..."
        alb_sg_id=$(aws ec2 create-security-group \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-name $alb_sg_name \
            --description "ALB for mywebsite" \
            --vpc-id $vpc_id \
            --output text --query 'GroupId')
        
        aws ec2 authorize-security-group-ingress \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-id $alb_sg_id \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 2>/dev/null || true
        
        aws ec2 authorize-security-group-ingress \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-id $alb_sg_id \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0 2>/dev/null || true
        
        log_info "  Created $alb_sg_name"
        ((created++)) || true
    else
        log_warn "  Security group $alb_sg_name already exists"
        ((exists++)) || true
    fi
    
    # ALB
    log_info "Checking ALB..."
    local alb_name="mywebsite-alb"
    local alb_arn=""
    local subnets
    subnets=$(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --filters "Name=vpc-id,Values=$vpc_id" --query 'Subnets[].SubnetId' --output text)
    
    if aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION --names $alb_name &>/dev/null; then
        alb_arn=$(aws elbv2 describe-load-balancers \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --names $alb_name \
            --output text --query 'LoadBalancers[0].LoadBalancerArn')
        log_warn "  ALB $alb_name already exists"
        ((exists++)) || true
    else
        log_info "  Creating ALB $alb_name..."
        alb_arn=$(aws elbv2 create-load-balancer \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --name $alb_name \
            --subnets $subnets \
            --security-groups $alb_sg_id \
            --scheme internet-facing \
            --type application \
            --output text --query 'LoadBalancers[0].LoadBalancerArn')
        log_info "  Created $alb_name"
        ((created++)) || true
    fi
    
    # Target Group
    log_info "Checking target group..."
    local tg_name="mywebsite-frontend-tg"
    local tg_arn=""
    
    if aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION --names $tg_name &>/dev/null; then
        tg_arn=$(aws elbv2 describe-target-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --names $tg_name \
            --output text --query 'TargetGroups[0].TargetGroupArn')
        log_warn "  Target group $tg_name already exists"
        ((exists++)) || true
    else
        log_info "  Creating target group $tg_name..."
        tg_arn=$(aws elbv2 create-target-group \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --name $tg_name \
            --protocol HTTP \
            --port 80 \
            --target-type ip \
            --vpc-id $vpc_id \
            --health-check-path / \
            --health-check-protocol HTTP \
            --health-check-interval-seconds 30 \
            --health-check-timeout-seconds 10 \
            --healthy-threshold-count 3 \
            --unhealthy-threshold-count 3 \
            --output text --query 'TargetGroups[0].TargetGroupArn')
        log_info "  Created $tg_name"
        ((created++)) || true
    fi
    
    # HTTP Listener
    log_info "Checking HTTP listener..."
    local http_listener_arn=""
    http_listener_arn=$(aws elbv2 describe-listeners \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --load-balancer-arn $alb_arn \
        --query 'Listeners[?Port==`80`].ListenerArn' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$http_listener_arn" ] && [ "$http_listener_arn" != "None" ]; then
        log_warn "  HTTP listener already exists"
        ((exists++)) || true
    else
        log_info "  Creating HTTP listener (port 80)..."
        http_listener_arn=$(aws elbv2 create-listener \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --protocol HTTP \
            --port 80 \
            --default-action TargetGroupArn=$tg_arn,Type=forward \
            --output text --query 'Listeners[0].ListenerArn')
        log_info "  Created HTTP listener"
        ((created++)) || true
    fi
    # ACM Certificate
    log_info "Checking ACM certificate..."
    local cert_arn="arn:aws:acm:$AWS_REGION:$ACCOUNT_ID:certificate/3ed4d5a9-167a-47de-ae94-c9a14d83a3b8"
    if aws acm describe-certificate --profile $AWS_PROFILE --region $AWS_REGION --certificate-arn "$cert_arn" &>/dev/null; then
        log_warn "  Certificate already exists"
        ((exists++)) || true
    else
        log_info "  Requesting ACM certificate..."
        cert_arn=$(aws acm request-certificate \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --domain-name reubenhow.com \
            --domain-name www.reubenhow.com \
            --validation-method DNS \
            --output text --query 'CertificateArn')
        log_warn "  Certificate requested - MUST validate in Cloudflare"
        ((created++)) || true
    fi
    
    # HTTPS Listener
    log_info "Checking HTTPS listener..."
    local https_listener_arn=""
    https_listener_arn=$(aws elbv2 describe-listeners \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --load-balancer-arn $alb_arn \
        --query 'Listeners[?Port==`443`].ListenerArn' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$https_listener_arn" ] && [ "$https_listener_arn" != "None" ]; then
        log_warn "  HTTPS listener already exists"
        ((exists++)) || true
    else
        log_info "  Creating HTTPS listener (port 443)..."
        https_listener_arn=$(aws elbv2 create-listener \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --load-balancer-arn $alb_arn \
            --protocol HTTPS \
            --port 443 \
            --certificates CertificateArn=$cert_arn \
            --default-action TargetGroupArn=$tg_arn,Type=forward \
            --ssl-policy ELBSecurityPolicy-2016-08 \
            --output text --query 'Listeners[0].ListenerArn')
        log_info "  Created HTTPS listener"
        ((created++)) || true
    fi
    
    # Get ALB DNS
    local alb_dns
    alb_dns=$(aws elbv2 describe-load-balancers \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --names $alb_name \
        --output text --query 'LoadBalancers[0].DNSName')
    echo "$alb_dns" > /tmp/alb-dns.txt
    
    log_info "AWS resources setup complete!"
    log_info "Summary: $created created, $exists already existed"
}

###############################################################################
# BUILD & PUSH IMAGES
###############################################################################
build_images() {
    log_info "=== Building Docker images ==="
    cd "$(dirname "$0")/.."
    
    # Login to ECR
    aws ecr get-login-password --profile $AWS_PROFILE | docker login --username AWS --password-stdin $REGISTRY
    
    # Frontend
    log_info "Building frontend..."
    docker build -t "$REGISTRY/frontend:$IMAGE_TAG" \
        --build-arg NODE_VERSION=22 \
        -f deployment/Dockerfile.frontend .
    
    # Backend
    log_info "Building backend..."
    docker build -t "$REGISTRY/backend:$IMAGE_TAG" \
        --build-arg PYTHON_VERSION=3.9-slim \
        -f deployment/Dockerfile.flask .
    
    # .NET API
    log_info "Building dotnet-api..."
    docker build -t "$REGISTRY/dotnet-api:$IMAGE_TAG" \
        --build-arg DOTNET_VERSION=8.0 \
        -f deployment/Dockerfile.dotnet .
    
    log_info "All images built!"
}

push_images() {
    log_info "=== Pushing images to ECR ==="
    
    for service in frontend backend dotnet-api; do
        log_info "Pushing $service..."
        docker push "$REGISTRY/$service:$IMAGE_TAG"
        docker tag "$REGISTRY/$service:$IMAGE_TAG" "$REGISTRY/$service:latest"
        docker push "$REGISTRY/$service:latest"
    done
    
    log_info "All images pushed!"
    log_info "Image tag: $IMAGE_TAG"
}

###############################################################################
# DEPLOY TO ECS
###############################################################################
deploy_to_ecs() {
    log_info "=== Deploying to ECS ==="
    
    # Update task definition with new image tags
    log_info "Registering new task definition..."
    
    local task_def_file="deployment/ecs-task-definition.json"
    local temp_file="/tmp/task-def-temp.json"
    
    # Get account ID from AWS CLI (not hardcoded)
    local ACCOUNT_ID
    ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query 'Account' --output text 2>/dev/null)
    log_info "Using AWS Account ID: $ACCOUNT_ID"
    
    # Update task definition: replace ACCOUNT_ID placeholder and :latest tags
    python3 -c "
import json, sys, re
with open('$task_def_file') as f:
    td = json.load(f)
tag = '$IMAGE_TAG'
account = '$ACCOUNT_ID'
# Replace account ID placeholder
for key in ['executionRoleArn', 'taskRoleArn']:
    if key in td:
        td[key] = td[key].replace('ACCOUNT_ID_PLACEHOLDER', account)
# Replace account ID in images and upgrade :latest to :<hash>
for i in range(len(td['containerDefinitions'])):
    img = td['containerDefinitions'][i]['image']
    img = img.replace('ACCOUNT_ID_PLACEHOLDER', account)
    img = re.sub(':latest$', ':' + tag, img)
    td['containerDefinitions'][i]['image'] = img
with open('$temp_file', 'w') as f:
    json.dump(td, f, indent=2)
"
    
    local task_def_arn
    task_def_arn=$(aws ecs register-task-definition \
        --profile $AWS_PROFILE \
        --cli-input-json file://"$temp_file" \
        --region $AWS_REGION \
        --output text --query 'taskDefinition.taskDefinitionArn')
    
    rm -f "$temp_file"
    
    log_info "Task definition registered: $task_def_arn"
    
    # Get VPC ID and subnets for service
    local vpc_id
    vpc_id=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
    local subnets
    subnets=$(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --filters "Name=vpc-id,Values=$vpc_id" --query 'Subnets[].SubnetId' --output text)
    
    # Get ALB security group ID
    local alb_sg_id
    alb_sg_id=$(aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION \
        --filters "Name=group-name,Values=mywebsite-alb-sg" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "")
    
    # Get ALB ARN
    local alb_arn
    alb_arn=$(aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION \
        --names mywebsite-alb \
        --output text --query 'LoadBalancers[0].LoadBalancerArn')
    
    # Get target group ARN
    local tg_arn
    tg_arn=$(aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION \
        --names mywebsite-frontend-tg \
        --output text --query 'TargetGroups[0].TargetGroupArn')
    
    # Create ECS service if it doesn't exist
    local service_exists
    service_exists=$(aws ecs describe-services --profile $AWS_PROFILE --cluster $CLUSTER --services $SERVICE --region $AWS_REGION --output json 2>/dev/null | python3 -c "import json,sys; data=json.load(sys.stdin); print('yes' if len(data.get('services',[])) > 0 else 'no')" 2>/dev/null || echo "no")
    
    if [ "$service_exists" = "no" ]; then
        log_info "Creating ECS service..."
        # Create network config JSON
        local subnet_json
        subnet_json=$(echo "$subnets" | tr '\t\n' ' ' | sed 's/ *$//' | tr -s ' ' | sed 's/^ *//;s/ /","/g;s/^/"/;s/$/"/')
        echo '{"awsvpcConfiguration":{"subnets":['$subnet_json'],"securityGroups":["'$alb_sg_id'"],"assignPublicIp":"ENABLED"}}' > /tmp/network-config.json

        
        aws ecs create-service \
            --profile $AWS_PROFILE \
            --cluster $CLUSTER \
            --service-name $SERVICE \
            --task-definition "$task_def_arn" \
            --desired-count 1 \
            --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
            --network-configuration file:///tmp/network-config.json \
            --load-balancers "targetGroupArn=$tg_arn,containerName=frontend,containerPort=80" \
            --region $AWS_REGION \
            --output text --query 'service.serviceArn'
        log_info "ECS service created!"
    else
        log_warn "ECS service already exists"
    fi
    
    # Update ECS service
    log_info "Updating ECS service (ALB targets auto-managed)..."
    aws ecs update-service \
        --profile $AWS_PROFILE \
        --cluster $CLUSTER \
        --service $SERVICE \
        --task-definition "$task_def_arn" \
        --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
        --force-new-deployment \
        --region $AWS_REGION \
        --output text \
        --query 'service.serviceName'
    
    log_info "Deployment initiated! ECS will automatically update ALB targets."
    log_info "Wait 60-90 seconds for new task to start and pass health checks."
}

###############################################################################
# CLOUDFLARE INSTRUCTIONS
###############################################################################
print_cloudflare_instructions() {
    echo ""
    echo "=============================================="
    echo "  CLOUDFLARE DNS SETUP REQUIRED"
    echo "=============================================="
    echo ""
    echo "1. Get ALB DNS:"
    echo "   cat /tmp/alb-dns.txt"
    echo ""
    echo "2. Add these records in Cloudflare:"
    echo ""
    echo "   ACM Validation (CNAME - proxy OFF/gray cloud):"
    echo "   Name: _eea0371d08f21d625e4ffc50c2b7bc1d.www"
    echo "   Value: _a74c7301325047431fd9465685dc3482.jkddzztszm.acm-validations.aws."
    echo ""
    echo "   Domain Records (proxy ON/orange cloud):"
    echo "   CNAME: www -> $(cat /tmp/alb-dns.txt 2>/dev/null || echo 'ALB-DNS')"
    echo "   CNAME: @   -> $(cat /tmp/alb-dns.txt 2>/dev/null || echo 'ALB-DNS')"
    echo ""
    echo "3. Wait 5-10 minutes for DNS propagation"
    echo "4. Test: https://reubenhow.com"
    echo "=============================================="
    echo ""
}

###############################################################################
# MAIN
###############################################################################
case "${1:-all}" in
    setup)
        check_prerequisites
        setup_all
        print_cloudflare_instructions
        ;;
    build)
        check_prerequisites
        build_images
        push_images
        ;;
    deploy)
        check_prerequisites
        deploy_to_ecs
        ;;
    all|*)
        check_prerequisites
        setup_all
        build_images
        push_images
        deploy_to_ecs
        print_cloudflare_instructions
        echo ""
        log_info "=== Deployment complete! ==="
        ;;
esac
