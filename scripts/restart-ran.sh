#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="open5gs"
DN_TARGET="10.10.6.100"
DN_SUBNET="10.10.6.0/24"
TUN_WAIT_TIMEOUT=60

echo "=== Restarting 5G Core + RAN ==="

echo "[1/7] Restarting SMF1 and AMF (clear stale session state)..."
kubectl rollout restart deployment -n "$NAMESPACE" open5gs-smf1
kubectl rollout restart deployment -n "$NAMESPACE" open5gs-amf
kubectl rollout status deployment -n "$NAMESPACE" open5gs-smf1 --timeout=120s
kubectl rollout status deployment -n "$NAMESPACE" open5gs-amf --timeout=120s

echo "[2/7] Deleting gNB and UE pods..."
kubectl delete pod -n "$NAMESPACE" -l component=gnb --wait=true
kubectl delete pod -n "$NAMESPACE" -l component=ue  --wait=true

echo "[3/7] Waiting for gNB..."
kubectl wait --for=condition=ready pod -n "$NAMESPACE" -l component=gnb --timeout=120s

echo "[4/7] Waiting for UE1..."
kubectl wait --for=condition=ready pod -n "$NAMESPACE" -l name=ue1 --timeout=120s

UE1_POD=$(kubectl get pod -n "$NAMESPACE" -l name=ue1 -o jsonpath='{.items[0].metadata.name}')

echo "[5/7] Waiting for uesimtun0 on $UE1_POD (up to ${TUN_WAIT_TIMEOUT}s)..."
SECONDS_WAITED=0
until kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- ip link show uesimtun0 &>/dev/null; do
  if [ "$SECONDS_WAITED" -ge "$TUN_WAIT_TIMEOUT" ]; then
    echo "ERROR: uesimtun0 did not appear within ${TUN_WAIT_TIMEOUT}s"
    exit 1
  fi
  sleep 2
  SECONDS_WAITED=$((SECONDS_WAITED + 2))
  echo "  ...waiting ($SECONDS_WAITED/${TUN_WAIT_TIMEOUT}s)"
done
kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- ip addr show uesimtun0

echo "[6/7] Ping test through 5G tunnel to $DN_TARGET..."
kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- sh -c "
ip route add $DN_SUBNET dev uesimtun0 2>/dev/null || true
ping -c 3 -W 2 $DN_TARGET
"

echo "[7/7] Scaling UPF1 back to 1 replica..."
kubectl scale deployment -n "$NAMESPACE" open5gs-upf1 --replicas=1
kubectl wait --for=condition=ready pod -n "$NAMESPACE" -l app.kubernetes.io/name=open5gs-upf1 --timeout=60s

echo ""
echo "=== Restart complete. 5G tunnel is healthy. UPF1 at 1 replica. ==="
