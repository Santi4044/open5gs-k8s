#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="open5gs"
DN_TARGET="10.10.6.100"
DN_SUBNET="10.10.6.0/24"

echo "=== Restarting RAN (gNB + UEs) ==="

echo "[1/5] Deleting gNB and UE pods..."
kubectl delete pod -n "$NAMESPACE" -l component=gnb --wait=true
kubectl delete pod -n "$NAMESPACE" -l component=ue  --wait=true

echo "[2/5] Waiting for gNB..."
kubectl wait --for=condition=ready pod -n "$NAMESPACE" -l component=gnb --timeout=120s

echo "[3/5] Waiting for UE1..."
kubectl wait --for=condition=ready pod -n "$NAMESPACE" -l name=ue1 --timeout=120s
sleep 5  # allow PDU session to fully establish

UE1_POD=$(kubectl get pod -n "$NAMESPACE" -l name=ue1 -o jsonpath='{.items[0].metadata.name}')

echo "[4/5] Checking uesimtun0 on $UE1_POD..."
kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- ip addr show uesimtun0

echo "[5/5] Ping test through 5G tunnel to $DN_TARGET..."
kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- sh -c "
ip route add $DN_SUBNET dev uesimtun0 2>/dev/null || true
ping -c 3 -W 2 $DN_TARGET
"

echo ""
echo "=== RAN restart complete. 5G tunnel is healthy. ==="
