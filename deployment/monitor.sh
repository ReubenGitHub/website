#!/bin/bash
# Monitor MyWebsite deployment
#
# Usage:
#   ./monitor.sh health     - Check ALB target health and ECS task status
#   ./monitor.sh logs       - View CloudWatch logs for all containers
#   ./monitor.sh all        - Health check + logs (default)

set -e

# Disable AWS CLI pager to output everything at once
export AWS_PAGER=""

AWS_PROFILE="mywebsite-production"
AWS_REGION="eu-west-2"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

###############################################################################
# HEALTH CHECK
###############################################################################
check_health() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  DEPLOYMENT HEALTH CHECK${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    # Check ECS tasks
    log_info "Checking ECS tasks..."
    echo ""
    
    local tasks
    tasks=$(aws ecs list-tasks --profile $AWS_PROFILE --region $AWS_REGION --cluster mywebsite-production --query 'taskArns[]' --output text 2>/dev/null || echo "")
    
    if [ -z "$tasks" ]; then
        log_error "No tasks running!"
        return 1
    fi
    
    for task_arn in $tasks; do
        local task_id
        task_id=$(echo "$task_arn" | cut -d'/' -f2)
        
        echo -e "${GREEN}Task: $task_id${NC}"
        
        # Get task details
        aws ecs describe-tasks --profile $AWS_PROFILE --region $AWS_REGION \
            --cluster mywebsite-production \
            --tasks "$task_arn" \
            --query 'tasks[0].{Status:lastStatus,Desired:desiredStatus,StartedBy:startedBy,Network:networkBindings[0].hostPort,PrivateIP:privateIpAddress}' \
            --output table 2>/dev/null || echo "  No task details"
        
        # Check each container
        echo "  Containers:"
        aws ecs describe-tasks --profile $AWS_PROFILE --region $AWS_REGION \
            --cluster mywebsite-production \
            --tasks "$task_arn" \
            --query 'tasks[0].containers[*].{Name:name,Status:currentState,ExitCode:exitCode,Reason:reason}' \
            --output table 2>/dev/null || echo "    No container details"
        
        echo ""
    done
    
    # Check ALB target health
    log_info "Checking ALB target health..."
    echo ""
    
    local tg_arn
    tg_arn=$(aws elbv2 describe-target-groups --profile $AWS_PROFILE --region $AWS_REGION --names mywebsite-frontend-tg --query 'TargetGroups[0].TargetGroupArn' --output text)
    
    if [ -n "$tg_arn" ]; then
        echo "Target Group Health:"
        aws elbv2 describe-target-health --profile $AWS_PROFILE --region $AWS_REGION \
            --target-group-arn "$tg_arn" \
            --query 'TargetHealthDescriptions[].{Target:Target.Id,State:TargetHealth.State,Reason:TargetHealth.Reason}' \
            --output table 2>/dev/null || echo "  No target health data"
    fi
    
    echo ""
    log_info "Health check complete!"
}

###############################################################################
# LOGS
###############################################################################
view_logs() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  CLOUDWATCH LOGS${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo "Available log groups:"
    echo ""
    
    local log_groups
    log_groups=$(aws logs describe-log-groups --profile $AWS_PROFILE --query 'logGroups[?contains(logGroupName, `mywebsite`)].logGroupName' --output text 2>/dev/null || echo "")
    
    if [ -z "$log_groups" ]; then
        log_warn "No log groups found. Tasks may not have started yet."
        return
    fi
    
    echo "$log_groups" | while read -r log_group; do
        echo -e "${GREEN}$log_group${NC}"
        echo "----------------------------------------"
        
        # Get last 20 log events
        aws logs get-log-events --profile $AWS_PROFILE \
            --log-group-name "$log_group" \
            --limit 20 \
            --query 'events[].{Time:timestamp,Message:message}' \
            --output text 2>/dev/null | while read -r line; do
            echo "  $line"
        done
        
        echo ""
    done
    
    echo "To tail logs in real-time:"
    echo "  aws logs tail /mywebsite/frontend --follow --profile $AWS_PROFILE"
    echo "  aws logs tail /mywebsite/backend --follow --profile $AWS_PROFILE"
    echo "  aws logs tail /mywebsite/dotnet-api --follow --profile $AWS_PROFILE"
    echo ""
}

###############################################################################
# MAIN
###############################################################################
case "${1:-all}" in
    health)
        check_health
        ;;
    logs)
        view_logs
        ;;
    all|*)
        check_health
        echo ""
        view_logs
        ;;
esac
