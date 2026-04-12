#!/usr/bin/env bash
set -euo pipefail

NAMESPACE_OPEN5GS="${NAMESPACE_OPEN5GS:-open5gs}"

UE_POD_LABEL_SELECTOR="${UE_POD_LABEL_SELECTOR:-}"
UE_POD_NAME_PATTERN="${UE_POD_NAME_PATTERN:-ueransim-ue1-}"

# DN/N6 iperf server (real datapath)
SERVER_IP="${SERVER_IP:-10.10.6.100}"
SERVER_PORT="${SERVER_PORT:-5201}"

BITRATE="${BITRATE:-20M}"
DURATION="${DURATION:-40}"
PKT_LEN="${PKT_LEN:-1200}"

# Ensure this subnet is routed via the UE tunnel
DN_SUBNET="${DN_SUBNET:-10.10.6.0/24}"
UE_TUN_IF="${UE_TUN_IF:-uesimtun0}"

find_ue_pod() {
  if [[ -n "$UE_POD_LABEL_SELECTOR" ]]; then
    kubectl get pod -n "$NAMESPACE_OPEN5GS" -l "$UE_POD_LABEL_SELECTOR" -o jsonpath='{.items[0].metadata.name}'
    return
  fi
  kubectl get pods -n "$NAMESPACE_OPEN5GS" | awk -v p="$UE_POD_NAME_PATTERN" '$1 ~ p {print $1; exit}'
}

UE_POD="$(find_ue_pod)"
if [[ -z "${UE_POD:-}" ]]; then
  echo "ERROR: Could not find UE pod (pattern: $UE_POD_NAME_PATTERN, selector: $UE_POD_LABEL_SELECTOR) in ns $NAMESPACE_OPEN5GS" >&2
  exit 1
fi

echo "[traffic] $(date -Iseconds) UE_POD=$UE_POD -> SERVER=${SERVER_IP}:${SERVER_PORT} bitrate=${BITRATE} len=${PKT_LEN} dur=${DURATION}s" >&2

kubectl exec -n "$NAMESPACE_OPEN5GS" "$UE_POD" -c ue -- sh -lc "
ip route add ${DN_SUBNET} dev ${UE_TUN_IF} 2>/dev/null || true
ip route get ${SERVER_IP} || true
iperf3 -u -c ${SERVER_IP} -p ${SERVER_PORT} -b ${BITRATE} -l ${PKT_LEN} -t ${DURATION} --connect-timeout 5000 || true
"
