#!/usr/bin/env bash
set -uo pipefail

NAMESPACE="open5gs"
DN_TARGET="10.10.6.100"
DN_SUBNET="10.10.6.0/24"
PASS=0; FAIL=0

check() {
  local desc="$1"; shift
  if "$@" > /dev/null 2>&1; then
    echo "  ✅ $desc"
    ((PASS++))
  else
    echo "  ❌ $desc"
    ((FAIL++))
  fi
}

echo "=== 5G Datapath Validation ==="
echo ""

echo "--- Core Network ---"
for comp in amf ausf bsf nrf nssf pcf scp smf1 smf2 udm udr; do
  check "open5gs-$comp running" \
    kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name="open5gs-$comp" -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running
done

echo ""
echo "--- UPF ---"
check "open5gs-upf1 running" \
  kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=open5gs-upf1 --field-selector=status.phase=Running -o name | grep -q .
check "UPF1 can ping iperf3 server ($DN_TARGET)" \
  kubectl exec -n "$NAMESPACE" $(kubectl get pod -n "$NAMESPACE" -o name | grep upf1 | head -1) -- ping -c 1 -W 2 "$DN_TARGET"

echo ""
echo "--- RAN ---"
check "gNB running" \
  kubectl get pod -n "$NAMESPACE" -l component=gnb --field-selector=status.phase=Running -o name | grep -q .

UE1_POD=$(kubectl get pod -n "$NAMESPACE" -l name=ue1 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
check "UE1 running" test -n "$UE1_POD"
check "UE1 uesimtun0 up" \
  kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- ip addr show uesimtun0 2>/dev/null
check "UE1 ping through tunnel to $DN_TARGET" \
  kubectl exec -n "$NAMESPACE" "$UE1_POD" -c ue -- sh -c "ip route add $DN_SUBNET dev uesimtun0 2>/dev/null||true; ping -c 1 -W 3 $DN_TARGET"

echo ""
echo "--- iperf3 Server ---"
check "iperf3-n6-server running" \
  kubectl get pod -n "$NAMESPACE" -l app=iperf3-n6-server --field-selector=status.phase=Running -o name | grep -q .
check "iperf3 server has N6 IP $DN_TARGET" \
  kubectl get pod -n "$NAMESPACE" -l app=iperf3-n6-server -o jsonpath='{.items[0].metadata.annotations.k8s\.v1\.cni\.cncf\.io/network-status}' | grep -q "$DN_TARGET"

echo ""
echo "--- HPA & Monitoring ---"
check "HPA open5gs-upf1-pps active" \
  kubectl get hpa -n "$NAMESPACE" open5gs-upf1-pps -o name
check "Prometheus scraping UPF1 metrics" \
  kubectl run -n monitoring --rm -i promtest-$RANDOM --restart=Never --image=curlimages/curl:8.5.0 -- \
  curl -fsS 'http://kps-kube-prometheus-stack-prometheus.monitoring.svc:9090/api/v1/query' \
  --data-urlencode 'query=up{service="open5gs-upf1-metrics"}' 2>/dev/null | grep -q '"1"'

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
