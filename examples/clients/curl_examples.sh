#!/usr/bin/env bash
# ==============================================================================
# S1stem cURL Cheat Sheet (Frictionless CLI Testing)
# ==============================================================================

SERVER_URL="http://localhost:8000"

echo "=== 1. Health & Server Status ==="
curl -s "${SERVER_URL}/health" | jq .

echo -e "\n=== 2. Direct Discrete Decision (/choice) ==="
curl -s -X POST "${SERVER_URL}/choice" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "High thermal alert in Reactor Core B. Temperature rising rapidly.",
    "actions": [
      { "id": "ENGAGE_COOLANT", "label": "Engage Coolant", "description": "Trigger emergency liquid nitrogen coolant flood." },
      { "id": "SCRAM_REACTOR",  "label": "SCRAM Reactor",  "description": "Drop control rods immediately for emergency shutdown." },
      { "id": "VENT_STEAM",     "label": "Vent Steam",     "description": "Release excess steam pressure to cooling towers." }
    ]
  }' | jq .

echo -e "\n=== 3. Register Action Space Once (/register-actions) ==="
curl -s -X POST "${SERVER_URL}/register-actions" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      { "id": "TOOL_FILE_READ",  "label": "read_file",  "description": "Read contents of a file on local filesystem." },
      { "id": "TOOL_FILE_WRITE", "label": "write_file", "description": "Create or write content to a file." },
      { "id": "TOOL_RUN_SHELL",  "label": "run_command","description": "Execute a bash or powershell terminal command." }
    ]
  }' | jq .

echo -e "\n=== 4. Fast Step Query Against Pre-cached Actions (/step) ==="
curl -s -X POST "${SERVER_URL}/step" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "List all active docker containers running on the system"
  }' | jq .
