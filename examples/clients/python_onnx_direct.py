#!/usr/bin/env python3
"""
S1stem In-Process ONNX Python Client (Pure Local Direct Inference)
==================================================================
Runs the standalone S1stem ONNX model (Xenova/bge-large-en-v1.5, 1024-dim)
directly inside your Python process without any HTTP server or Docker.

Requirements:
    pip install onnxruntime numpy tokenizers
    # or: pip install transformers onnxruntime numpy

Model Files:
    Requires the standalone model files in `models/s1stem-bge-large/`.
    Generate them anytime with:
        npm run model:download
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError:
    ort = None
    Tokenizer = None

class S1stemOnnxEngine:
    """
    Direct in-process System 1 Contrastive Language Model (CLM) Engine in Python.
    Implements the two-tower asymmetric CLM inner-product matching architecture.
    """
    def __init__(self, model_dir: str = "models/s1stem-bge-large", temperature: float = 0.04):
        if ort is None or Tokenizer is None:
            raise ImportError(
                "Missing required packages for direct ONNX execution.\n"
                "Please run: pip install onnxruntime numpy tokenizers"
            )

        self.model_dir = model_dir
        self.temperature = temperature
        
        onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model file not found at: {onnx_path}\n"
                f"Please run `npm run model:download` first to export the standalone model."
            )

        print(f"[s1stem-onnx] Loading ONNX runtime session: {onnx_path}")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(length=512)

        # Pre-cached Action Matrix A in R^(K x 1024)
        self.action_cache: List[Dict[str, Any]] = []
        self.action_matrix: Optional[np.ndarray] = None
        self.embedding_dim = 1024

    def _encode_text(self, text: str) -> np.ndarray:
        """Encodes text to a 1024-dimensional normalized latent vector."""
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]  # [1, seq_len, 1024]
        
        # [CLS] pooling (index 0)
        cls_embedding = last_hidden_state[:, 0, :]  # [1, 1024]
        
        # L2-normalization
        norm = np.linalg.norm(cls_embedding, ord=2, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return (cls_embedding / norm)[0]  # [1024]

    def register_actions(self, actions: List[Dict[str, Any]]) -> int:
        """
        One-Shot candidate action space registration.
        Encodes all candidate actions once into Action Matrix A in RAM.
        Scales to 3,000+ actions with zero prompt bloat.
        """
        if not actions:
            raise ValueError("actions list must not be empty.")

        t0 = time.perf_counter()
        k = len(actions)
        matrix = np.zeros((k, self.embedding_dim), dtype=np.float32)
        self.action_cache = []

        for i, act in enumerate(actions):
            act_id = str(act.get("id", f"act_{i}"))
            label = str(act.get("label", act_id))
            desc = str(act.get("description", ""))
            
            # Format action descriptor matching S1stem CLM specification
            desc_text = f"{label}. {desc}" if desc else label
            prefix = "" if act_id.isdigit() else f"{act_id}: "
            full_text = f"{prefix}{desc_text}".strip()

            vec = self._encode_text(full_text)
            matrix[i] = vec
            self.action_cache.append({
                "action": act,
                "id": act_id,
                "label": label,
                "description": desc
            })

        self.action_matrix = matrix
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[s1stem-onnx] Pre-cached {k} actions in {elapsed_ms:.1f} ms.")
        return k

    def choice(
        self,
        state: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        mask: Optional[List[bool]] = None,
        biases: Optional[List[float]] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Resolves the winning discrete decision for the given state.
        
        Input Contract:
          - state (str): Contextual observation, user intent, code error, or visual summary.
          - actions (list[dict]): Optional candidate actions. If passed, registers them.
          - mask (list[bool]): Safety veto mask (True = allow, False = block).
          - biases (list[float]): Policy logit biases beta_i.
          - temperature (float): Softmax temperature (default: 0.04).
        
        Output Contract:
          - winner (str): ID of winning action.
          - selectedAction (dict): Winning action descriptor.
          - confidence (float): Calibrated probability [0.0, 1.0].
          - ranking (list[dict]): Ranked action candidates sorted by probability.
          - latencyMs (float): Latency in milliseconds.
        """
        t0 = time.perf_counter()

        if actions is not None:
            self.register_actions(actions)

        if self.action_matrix is None or len(self.action_cache) == 0:
            raise ValueError("No actions registered. Provide actions or call register_actions() first.")

        temp = temperature if temperature is not None else self.temperature
        
        # 1. Encode incoming state vector v_s in R^1024
        v_s = self._encode_text(state)

        # 2. Fast Inner Product: logits = (v_s · A^T) / temp
        k = len(self.action_cache)
        logits = np.dot(self.action_matrix, v_s) / temp

        # 3. Apply Policy Biases
        if biases is not None:
            for i, b in enumerate(biases):
                if i < k:
                    logits[i] += b / temp

        # 4. Apply Safety Guardrail Mask (Hard Veto)
        if mask is not None:
            for i, m in enumerate(mask):
                if i < k and not m:
                    logits[i] = -1e9

        # 5. Numerically stable Softmax
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        probs = exp_logits / sum_exp if sum_exp > 0 else np.full(k, 1.0 / k)

        # 6. Rank candidates
        best_idx = int(np.argmax(probs))
        ranking = []
        for i in range(k):
            entry = self.action_cache[i]
            ranking.append({
                "id": entry["id"],
                "label": entry["label"],
                "probability": float(probs[i]),
                "action": entry["action"]
            })

        ranking.sort(key=lambda x: x["probability"], reverse=True)
        for rank_idx, item in enumerate(ranking):
            item["rank"] = rank_idx + 1

        latency_ms = (time.perf_counter() - t0) * 1000

        winner_entry = self.action_cache[best_idx]
        return {
            "winner": winner_entry["id"],
            "selectedAction": winner_entry["action"],
            "confidence": float(probs[best_idx]),
            "ranking": ranking,
            "latencyMs": latency_ms,
            "cachedActionsCount": k
        }


# --- Example Standalone Verification ---
if __name__ == "__main__":
    print("=" * 80)
    print("S1STEM DIRECT IN-PROCESS ONNX PYTHON EXAMPLE")
    print("=" * 80)

    # 1. Initialize Engine directly loading ONNX model files
    try:
        engine = S1stemOnnxEngine(model_dir="models/s1stem-bge-large", temperature=0.04)
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        print("Note: Ensure you have run `npm run model:download` and installed `pip install onnxruntime numpy tokenizers`.")
        exit(1)

    # 2. Define Action Contract
    candidate_actions = [
        {
            "id": "NAV_CORRIDOR_A",
            "label": "Corridor A (East)",
            "description": "Clear unobstructed hallway heading towards extraction point."
        },
        {
            "id": "NAV_CORRIDOR_B",
            "label": "Corridor B (North)",
            "description": "Narrow access shaft blocked by debris and toxic steam leak."
        },
        {
            "id": "HOLD_POSITION",
            "label": "Hold Position",
            "description": "Remain static and scan surroundings with radar."
        }
    ]

    # 3. One-Shot Action Registration
    engine.register_actions(candidate_actions)

    # 4. Resolve Runtime Decision
    observation = "Drone telemetry: Forward camera detects clear open path towards East waypoint. Radiation nominal."
    decision = engine.choice(
        state=observation,
        mask=[True, False, True],  # Corridor B is blocked by safety mask
        biases=[1.5, 0.0, 0.0]     # Extra bias toward direct navigation
    )

    print("\n--- Decision Result ---")
    print(f"Observation: {observation}")
    print(f"Winner:      {decision['winner']}")
    print(f"Confidence:  {decision['confidence'] * 100:.1f}%")
    print(f"Latency:     {decision['latencyMs']:.2f} ms")
    print("\nRanking:")
    for r in decision["ranking"]:
        print(f"  [#{r['rank']}] {r['id'].padEnd(16)} -> {r['probability'] * 100:.1f}%")
