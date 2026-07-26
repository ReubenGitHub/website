#!/bin/bash
# List all AWS resources for MyWebsite deployment
#
# Usage:
#   ./list-resources.sh              - List all resources
#   ./list-resources.sh ecr          - List ECR repositories only
#   ./list-resources.sh ecs          - List ECS clusters/services/tasks
#   ./list-resources.sh alb          - List ALBs and target groups
#   ./list-resources.sh acm          - List ACM certificates
#   ./list-resources.sh iam          - List IAM roles
#   ./list-resources.sh vpc          - List VPCs and subnets
#   ./list-resources.sh logs         - List CloudWatch log groups
#   ./list-resources.sh sg           - List security groups

set -e

# Disable AWS CLI pager to output everything at once
export AWS_PAGER=""

AWS_PROFILE="mywebsite-production"
AWS_REGION="eu-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query 'Account' --output text)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  MyWebsite AWS Resources - ${AWS_REGION}${NC}"
echo -e "${BLUE}  Account: ${ACCOUNT_ID}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

###############################################################################
# ECR REPOSITORIES
###############################################################################
echo -e "${GREEN}ECR REPOSITORIES${NC}"
echo "----------------------------------------"
aws ecr describe-repositories --profile $AWS_PROFILE --region $AWS_REGION --query 'repositories[].repositoryName' --output text 2>/dev/null | while read -r repo; do
    echo "  - $repo"
    
    # List images in each repo
    images=$(aws ecr list-images --profile $AWS_PROFILE --region $AWS_REGION --repository-name "$repo" --query 'imageIds[].imageTag' --output text 2>/dev/null || echo "")
    if [ -n "$images" ]; then
        echo "    Images:"
        echo "$images" | while read -r img; do
            echo "      - $img"
        done
    fi
done
echo ""

###############################################################################
# ECS CLUSTERS, SERVICES, TASKS
###############################################################################
echo -e "${GREEN}ECS CLUSTERS${NC}"
echo "----------------------------------------"
aws ecs describe-clusters --profile $AWS_PROFILE --region $AWS_REGION --clusters mywebsite-production --query 'clusters[0].{Name:clusterName,Status:clusterStatus,Tasks:activeTasksCount}' --output table 2>/dev/null || echo "  No clusters found"
echo ""

echo -e "${GREEN}ECS SERVICES${NC}"
echo "----------------------------------------"
aws ecs list-services --profile $AWS_PROFILE --region $AWS_REGION --cluster mywebsite-production --query 'serviceArns[]' --output text 2>/dev/null | while read -r service_arn; do
    service_name=$(echo "$service_arn" | cut -d'/' -f2)
    echo "  Service: $service_name"
    
    aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION \
        --cluster mywebsite-production \
        --services "$service_name" \
        --query 'services[0].{Running:runningCount,Desired:desiredCount,TaskDef:taskDefinition,LaunchType:launchType}' \
        --output table 2>/dev/null || echo "    No details"
    
    # List running tasks
    echo "    Tasks:"
    aws ecs list-tasks --profile $AWS_PROFILE --region $AWS_REGION \
        --cluster mywebsite-production \
        --service-name "$service_name" \
        --query 'taskArns[]' --output text 2>/dev/null | while read -r task_arn; do
        task_id=$(echo "$task_arn" | cut -d'/' -f2)
        echo "      - $task_id"
        
        # Get container instances
        aws ecs describe-tasks --profile $AWS_PROFILE --region $AWS_REGION \
            --cluster mywebsite-production \
            --tasks "$task_arn" \
            --query 'tasks[0].{StartedBy:startedBy,Containers:containers[0].name,LastStatus:lastStatus,Network:networkBindings[0].hostPort}' \
            --output table 2>/dev/null || echo "        No task details"
    done
done
echo ""

###############################################################################
# APPLICATION LOAD BALANCERS
###############################################################################
echo -e "${GREEN}APPLICATION LOAD BALANCERS${NC}"
echo "----------------------------------------"
aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION --query 'LoadBalancers[].{Name:Name,DNS:DNSName,State:state.name}' --output table 2>/dev/null || echo "  No ALBs found"
echo ""

echo -e "${GREEN}TARGET GROUPS${NC}"
echo "----------------------------------------"
aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION --query 'TargetGroups[?contains(TargetGroupName, `mywebsite`)]' --output table 2>/dev/null || echo "  No target groups found"
echo ""

echo -e "${GREEN}LISTENERS${NC}"
echo "----------------------------------------"
# Get ALB ARN first
alb_arn=$(aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION --names mywebsite-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo "")
if [ -n "$alb_arn" ]; then
    aws elbv2 describe-listeners --profile $AWS_PROFILE --region $AWS_REGION --load-balancer-arn "$alb_arn" --query 'Listeners[].{Port:Port,Protocol:Protocol,ListenerArn:ListenerArn}' --output table 2>/dev/null || echo "  No listeners found"
else
    echo "  No ALB found"
fi
echo ""

###############################################################################
# ACM CERTIFICATES
###############################################################################
echo -e "${GREEN}ACM CERTIFICATES${NC}"
echo "----------------------------------------"
aws acm list-certificates --profile $AWS_PROFILE --region $AWS_REGION --query 'CertificateSummaryList[?contains(DomainName, `reubenhow`)].{Domain:DomainName,Status:Status,Arn:CertificateArn}' --output table 2>/dev/null || echo "  No certificates found"
echo ""

###############################################################################
# IAM ROLES
###############################################################################
echo -e "${GREEN}IAM ROLES${NC}"
echo "----------------------------------------"
aws iam list-roles --profile $AWS_PROFILE --query 'Roles[?contains(RoleName, `ecs`)]' --output table 2>/dev/null || echo "  No ECS roles found"
echo ""

###############################################################################
# VPCS AND SUBNETS
###############################################################################
echo -e "${GREEN}VPCS${NC}"
echo "----------------------------------------"
vpc_id=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --vpc-ids $vpc_id --query 'Vpcs[].{ID:VpcId,Cidr:CidrBlock,Default:IsDefault}' --output table 2>/dev/null || echo "  No VPCs found"
echo ""

echo -e "${GREEN}SUBNETS${NC}"
echo "----------------------------------------"
vpc_id=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION \
    --filters "Name=vpc-id,Values=$vpc_id" \
    --query 'Subnets[].{ID:SubnetId,Cidr:CidrBlock,AZ:AvailabilityZone}' \
    --output table 2>/dev/null || echo "  No subnets found"
echo ""

###############################################################################
# SECURITY GROUPS
###############################################################################
echo -e "${GREEN}SECURITY GROUPS${NC}"
echo "----------------------------------------"
aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION \
    --filters "Name=vpc-id,Values=$vpc_id" \
    --query 'SecurityGroups[?contains(GroupName, `mywebsite`)]' \
    --output table 2>/dev/null || echo "  No security groups found"
echo ""

###############################################################################
# CLOUDWATCH LOG GROUPS
###############################################################################
echo -e "${GREEN}CLOUDWATCH LOG GROUPS${NC}"
echo "----------------------------------------"
aws logs describe-log-groups --profile $AWS_PROFILE --query 'logGroups[?contains(logGroupName, `mywebsite`)]' --output table 2>/dev/null || echo "  No log groups found"
echo ""

###############################################################################

###############################################################################
# TASK DEFINITIONS
###############################################################################
echo -e "${GREEN}TASK DEFINITIONS${NC}"
echo "----------------------------------------"
aws ecs list-task-definitions --profile $AWS_PROFILE --region $AWS_REGION --output text 2>/dev/null | tail -n +2 | grep mywebsite-production | while read -r _ arn; do
    revision=$(echo "$arn" | cut -d':' -f7)
    status=$(aws ecs describe-task-definition --profile $AWS_PROFILE --region $AWS_REGION --task-definition "$arn" --query 'taskDefinition.status' --output text 2>/dev/null)
    echo "  $arn (revision $revision, $status)"
done
count=$(aws ecs list-task-definitions --profile $AWS_PROFILE --region $AWS_REGION --output text 2>/dev/null | tail -n +2 | grep mywebsite-production | wc -l)
echo "  Total: $count task definition(s)"
echo ""

# SUMMARY
###############################################################################
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SUMMARY${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "  ECR Repos:       $(aws ecr describe-repositories --profile $AWS_PROFILE --region $AWS_REGION --query 'length(repositories)' --output text 2>/dev/null || echo 0)"
echo "  ECS Clusters:    $(aws ecs describe-clusters --profile $AWS_PROFILE --region $AWS_REGION --clusters mywebsite-production --query 'length(clusters)' --output text 2>/dev/null || echo 0)"
echo "  ECS Services:    $(aws ecs list-services --profile $AWS_PROFILE --region $AWS_REGION --cluster mywebsite-production --query 'length(serviceArns)' --output text 2>/dev/null || echo 0)"
echo "  ALBs:            $(aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION --query 'length(LoadBalancers)' --output text 2>/dev/null || echo 0)"
echo "  Target Groups:   $(aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION --query 'length(TargetGroups)' --output text 2>/dev/null || echo 0)"
ALB_ARN=$(aws elbv2 describe-load-balancers --profile $AWS_PROFILE --region $AWS_REGION --names mywebsite-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo "")
echo "  Listeners:       $(aws elbv2 describe-listeners --profile $AWS_PROFILE --region $AWS_REGION --load-balancer-arn "$ALB_ARN" --query 'length(Listeners)' --output text 2>/dev/null || echo 0)"
echo "  ACM Certs:       $(aws acm list-certificates --profile $AWS_PROFILE --region $AWS_REGION --query 'length(CertificateSummaryList)' --output text 2>/dev/null || echo 0)"
echo "  IAM Roles:       $(aws iam list-roles --profile $AWS_PROFILE --region $AWS_REGION --query 'length(Roles[?contains(RoleName, `ecs`)])' --output text 2>/dev/null || echo 0)"
VPC_ID=$(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")
echo "  VPCs:            $(aws ec2 describe-vpcs --profile $AWS_PROFILE --region $AWS_REGION --query 'length(Vpcs[?IsDefault==`true`])' --output text 2>/dev/null || echo 0)"
echo "  Subnets:         $(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --filters "Name=vpc-id,Values=$VPC_ID" --query 'length(Subnets)' --output text 2>/dev/null || echo 0)"
echo "  Security Groups: $(aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION --filters "Name=vpc-id,Values=$VPC_ID" --query 'length(SecurityGroups)' --output text 2>/dev/null || echo 0)"
echo "  Log Groups:      $(aws logs describe-log-groups --profile $AWS_PROFILE --region $AWS_REGION --query 'length(logGroups)' --output text 2>/dev/null || echo 0)"
echo "  Task Definitions: $(aws ecs list-task-definitions --profile $AWS_PROFILE --region $AWS_REGION --query 'length(taskDefinitionArns)' --output text 2>/dev/null || echo 0)"
echo ""
echo "  To check health:"
echo "    ./monitor.sh"
echo ""
echo "  To tear down everything:"
echo "    ./teardown.sh"
echo ""
