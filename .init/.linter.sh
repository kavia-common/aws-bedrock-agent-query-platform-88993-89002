#!/bin/bash
cd /home/kavia/workspace/code-generation/aws-bedrock-agent-query-platform-88993-89002/backend_aws_bedrock_rag
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

