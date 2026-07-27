#!/usr/bin/env bash
# AWS SSO Login Script
# Usage: source ./deployment/login.sh

set -e

echo "Logging into AWS SSO..."
aws sso login --profile mywebsite-production

echo "AWS SSO login successful!"
echo "Your temporary credentials are now active for the mywebsite-production profile."
