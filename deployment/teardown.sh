#!/bin/bash
# MyWebsite Deployment Teardown Script
# Cleans up all AWS resources created during deployment
# Usage: ./teardown.sh [--force]
# DANGER: This will delete all resources and they cannot be recovered!

set -e

# Disable AWS CLI pager to output everything at once
export AWS_PAGER=""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROFILE="mywebsite-production"
REGION="eu-west-2"
CLUSTER="mywebsite-production"
SERVICE="mywebsite-production"

# Check for force flag
FORCE=false
if [[ "${1}" == "--force" ]]; then
    FORCE=true
fi

# Confirm teardown
confirm_tearown() {
    if [ "$FORCE" = false ]; then
        log_warn "This will DELETE all AWS resources!"
        echo ""
        echo "Resources to be deleted:"
        echo "  - ECS cluster and service"
        echo "  - ECR repositories and images"
        echo "  - IAM roles"
        echo "  - CloudWatch log groups"
        echo "  - Security group"
        echo ""
        read -p "Type 'DELETE EVERYTHING' to confirm: " confirmation
        if [ "$confirmation" != "DELETE EVERYTHING" ]; then
            log_info "Teardown cancelled"
            exit 0
        fi
    fi
}

# Stop ECS service first
stop_ecs_service() {
    log_info "Stopping ECS service..."
    
    aws ecs update-service \
        --profile $PROFILE \
        --region $REGION \
        --cluster $CLUSTER \
        --service $SERVICE \
        --desired-count 0 \
        --output json >/dev/null 2>&1 || true
    
    log_info "ECS service stopped"
}

# Delete ECS service
delete_ecs_service() {
    log_info "Deleting ECS service..."
    
    aws ecs delete-service \
        --profile $PROFILE \
        --region $REGION \
        --cluster $CLUSTER \
        --service $SERVICE \
        --force \
        --output json >/dev/null 2>&1 || true
    
    log_info "ECS service deleted"
}

# Delete ECS cluster
delete_ecs_cluster() {
    log_info "Deleting ECS cluster..."
    
    aws ecs delete-cluster \
        --profile $PROFILE \
        --region $REGION \
        --cluster $CLUSTER \
        --output json >/dev/null 2>&1 || true
    
    log_info "ECS cluster deleted"
}

# Delete ECR repositories (and force delete images)
delete_ecr_repositories() {
    log_info "Deleting ECR repositories..."
    
    local repositories=("frontend" "backend" "dotnet-api")
    
    for repo in "${repositories[@]}"; do
        log_info "Deleting repository: reubenhow/${repo}"
        
        # Force delete all images first (try twice to handle pagination)
        aws ecr delete-image \
            --profile $PROFILE \
            --repository-name "reubenhow/${repo}" \
            --image-id imageTag=latest \
            --output json >/dev/null 2>&1 || true
        
        aws ecr delete-image \
            --profile $PROFILE \
            --repository-name "reubenhow/${repo}" \
            --image-id imageTag=latest \
            --output json >/dev/null 2>&1 || true
        
        # Delete repository
        aws ecr delete-repository \
            --profile $PROFILE \
            --repository-name "reubenhow/${repo}" \
            --force \
            --output json >/dev/null 2>&1 || true
    done
    
    log_info "ECR repositories deleted"
}

# Delete IAM roles
delete_iam_roles() {
    log_info "Deleting IAM roles..."
    
    local roles=("ecsTaskRole" "ecsTaskExecutionRole")
    
    for role in "${roles[@]}"; do
        # Detach all policies
        policies=$(aws iam list-attached-role-policies \
            --profile $PROFILE \
            --role-name $role \
            --query 'AttachedPolicies[*].PolicyArn' \
            --output text 2>/dev/null || echo "")
        
        for policy in $policies; do
            aws iam detach-role-policy \
                --profile $PROFILE \
                --role-name $role \
                --policy-arn $policy \
                --output json >/dev/null 2>&1 || true
        done
        
        # Delete role
        aws iam delete-role \
            --profile $PROFILE \
            --role-name $role \
            --output json >/dev/null 2>&1 || true
    done
    
    log_info "IAM roles deleted"
}

# Delete CloudWatch log groups
delete_cloudwatch_logs() {
    log_info "Deleting CloudWatch log groups..."
    
    local log_groups=("/mywebsite/frontend" "/mywebsite/backend" "/mywebsite/dotnet-api")
    
    for log_group in "${log_groups[@]}"; do
        aws logs delete-log-group \
            --profile $PROFILE \
            --log-group-name $log_group \
            --output json >/dev/null 2>&1 || true
    done
    
    log_info "CloudWatch log groups deleted"
}

# Delete all mywebsite security groups
delete_security_groups() {
    log_info "Deleting all mywebsite security groups..."
    
    # Find all security groups with mywebsite in the name
    local sg_ids
    sg_ids=$(aws ec2 describe-security-groups \
        --profile $PROFILE \
        --region $REGION \
        --filters "Name=group-name,Values=mywebsite*" \
        --query 'SecurityGroups[*].GroupId' \
        --output text 2>/dev/null || echo "")
    
    for sg_id in $sg_ids; do
        log_info "Deleting security group: $sg_id"
        
        # Try to remove ingress rules first
        aws ec2 revoke-security-group-ingress \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 \
            --output json >/dev/null 2>&1 || true
        
        aws ec2 revoke-security-group-ingress \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0 \
            --output json >/dev/null 2>&1 || true
        
        aws ec2 revoke-security-group-ingress \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 8080 \
            --cidr 0.0.0.0/0 \
            --output json >/dev/null 2>&1 || true
        
        aws ec2 revoke-security-group-ingress \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 5000 \
            --cidr 0.0.0.0/0 \
            --output json >/dev/null 2>&1 || true
        
        aws ec2 revoke-security-group-ingress \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 5001 \
            --cidr 0.0.0.0/0 \
            --output json >/dev/null 2>&1 || true
        
        # Delete the security group
        aws ec2 delete-security-group \
            --profile $PROFILE \
            --region $REGION \
            --group-id "$sg_id" \
            --output json >/dev/null 2>&1 || true
    done
    
    log_info "All security groups deleted"
}

# Delete ALB listeners (must be done before deleting ALB)
delete_listeners() {
    log_info "Deleting ALB listeners..."
    
    local alb_arn
    alb_arn=$(aws elbv2 describe-load-balancers \
        --profile $PROFILE \
        --region $REGION \
        --names mywebsite-alb \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$alb_arn" ] && [ "$alb_arn" != "None" ]; then
        # Get all listener ARNs
        local listeners
        listeners=$(aws elbv2 describe-listeners \
            --profile $PROFILE \
            --region $REGION \
            --load-balancer-arn "$alb_arn" \
            --query 'Listeners[*].ListenerArn' \
            --output text 2>/dev/null || echo "")
        
        for listener in $listeners; do
            log_info "Deleting listener: $listener"
            aws elbv2 delete-listener \
                --profile $PROFILE \
                --region $REGION \
                --listener-arn "$listener" \
                --output json >/dev/null 2>&1 || true
        done
    fi
    
    log_info "Listeners deleted"
}

# Delete target group
delete_target_group() {
    log_info "Deleting target group..."
    
    local tg_arn
    tg_arn=$(aws elbv2 describe-target-groups \
        --profile $PROFILE \
        --region $REGION \
        --names mywebsite-frontend-tg \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$tg_arn" ] && [ "$tg_arn" != "None" ]; then
        aws elbv2 delete-target-group \
            --profile $PROFILE \
            --region $REGION \
            --target-group-arn "$tg_arn" \
            --output json >/dev/null 2>&1 || true
    fi
    
    log_info "Target group deleted"
}

# Delete ALB
delete_alb() {
    log_info "Deleting ALB..."
    
    # Find ALB by name instead of hardcoded ARN
    local alb_arn
    alb_arn=$(aws elbv2 describe-load-balancers \
        --profile $PROFILE \
        --region $REGION \
        --names mywebsite-alb \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$alb_arn" ] && [ "$alb_arn" != "None" ]; then
        aws elbv2 delete-load-balancer \
            --profile $PROFILE \
            --region $REGION \
            --load-balancer-arn "$alb_arn" \
            --output json >/dev/null 2>&1 || true
    fi
    
    log_info "ALB deleted"
}

# Delete ACM certificate
delete_acm_certificate() {
    log_info "Deleting ACM certificate..."
    
    local ACCOUNT_ID
    ACCOUNT_ID=$(aws sts get-caller-identity --profile $PROFILE --query 'Account' --output text 2>/dev/null)
    local cert_arn="arn:aws:acm:eu-west-2:${ACCOUNT_ID}:certificate/3ed4d5a9-167a-47de-ae94-c9a14d83a3b8"
    
    aws acm delete-certificate \
        --profile $PROFILE \
        --region $REGION \
        --certificate-arn "$cert_arn" \
        --output json >/dev/null 2>&1 || true
    
    log_info "ACM certificate deleted"
}

# Main teardown flow
main() {
    log_info "MyWebsite Production Teardown"
    log_info "=============================="
    echo ""
    
    confirm_tearown
    
    log_info "Starting teardown..."
    echo ""
    
    # Order matters: stop tasks first, then delete resources
    stop_ecs_service
    delete_ecs_service
    delete_ecs_cluster
    delete_listeners
    delete_target_group
    delete_alb
    delete_acm_certificate
    delete_ecr_repositories
    delete_iam_roles
    delete_cloudwatch_logs
    delete_security_groups
    
    echo ""
    log_info "=========================================="
    log_info "Teardown Complete!"
    log_info "=========================================="
    echo ""
    log_warn "All resources have been deleted"
    log_warn "You will need to re-run deploy.sh for next deployment"
}

# Run main function
main
