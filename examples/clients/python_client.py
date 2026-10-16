#!/usr/bin/env python3
"""
S1stem Python Client (Frictionless System 1 Discrete Decisions)
==============================================================
Zero-dependency client connecting Python agents (LangChain, AutoGen, CrewAI,
LlamaIndex, custom robotics/game loops) to a local S1stem container/server.

Prerequisites:
    pip install urllib3  # Or use standard library urllib.request (zero install)
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

class S1stemClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> Dict[str, Any]:
        """Check S1stem server status and model metadata."""
        req = urllib.request.Request(f"{self.base_url}/health")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def choice(
        self,
        state: str,
        actions: Optional[List[Dict[str, str]]] = None,
        mask: Optional[List[bool]] = None,
        biases: Optional[List[float]] = None,
        temperature: float = 0.04
    ) -> Dict[str, Any]:
        """
        Evaluate candidate actions for a given state observation.
        
        Args:
            state: Textual or visual prompt describing current environment/state.
            actions: List of candidates [{'id': 'TOOL_1', 'label': 'Search', 'description': '...'}]
            mask: Optional boolean mask (True = allow, False = veto/mask out)
            biases: Optional additive logit bias vector beta_i
            temperature: Softmax temperature tau (default: 0.04)
        """
        payload = {"state": state, "temperature": temperature}
        if actions:
            payload["actions"] = actions
        if mask:
            payload["mask"] = mask
        if biases:
            payload["biases"] = biases
        return self._post("/choice", payload)

    def register_actions(self, actions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Pre-cache action embeddings in RAM once for ultra-fast O(1) query matching."""
        return self._post("/register-actions", {"actions": actions})

    def step(
        self,
        state: str,
        mask: Optional[List[bool]] = None,
        biases: Optional[List[float]] = None,
        temperature: float = 0.04
    ) -> Dict[str, Any]:
        """O(1) sub-millisecond evaluation against pre-cached actions."""
        payload = {"state": state, "temperature": temperature}
        if mask:
            payload["mask"] = mask
        if biases:
            payload["biases"] = biases
        return self._post("/step", payload)


# --- Example Usage ---
if __name__ == "__main__":
    client = S1stemClient("http://localhost:8000")
    print(f"Connecting to S1stem at {client.base_url}...")
    
    try:
        health = client.health()
        print(f"✓ S1stem Online: {health['engine']} (dim: {health['embeddingDim']}, offline: {health['isOfflineModel']})")
    except Exception as e:
        print(f"⚠️ Could not connect to S1stem: {e}")
        print("  Make sure the container or server is running (e.g. `docker compose up -d` or `npm run serve:api`).")
        exit(1)

    # 1. Direct Discrete Decision
    result = client.choice(
        state="Intruder alarm tripped at perimeter Sector 4. Cameras show unverified movement.",
        actions=[
            {"id": "LOCKDOWN_SECTOR", "label": "Lockdown Sector", "description": "Isolate perimeter gates and sound sirens."},
            {"id": "DISPATCH_DRONE",   "label": "Dispatch Drone",   "description": "Send reconnaissance drone for immediate aerial verification."},
            {"id": "IGNORE_ALARM",     "label": "Ignore Alarm",     "description": "Dismiss alert as sensor glitch."}
        ]
    )

    print("\n--- Decision Result ---")
    print(f"Winner:     {result['winner']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Latency:    {result['latencyMs']:.2f} ms")
    for rank in result["ranking"]:
        print(f"  • {rank['id']}: {rank['probability'] * 100:.1f}%")
