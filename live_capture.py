"""
live_capture.py
---------------
REAL real-time traffic source for the NIDS dashboard.

Sniffs live packets off a network interface with scapy, assembles them into
connections (flows), computes the NSL-KDD *traffic* features per connection, and
appends one CSV line per finished connection to data/stream.csv — the same file
the Streamlit "Real-time Monitoring" page tails.

IMPORTANT — honest limitations
-------------------------------
NSL-KDD has 41 features. Only the ~20 "traffic" features are derivable from
packet headers (bytes, durations, per-host/per-service counts and error rates).
The ~21 "content" features (hot, num_failed_logins, logged_in, su_attempted,
root_shell, ...) require payload / host-OS inspection and CANNOT be obtained
from sniffing — they are zero-filled here. Practical effect: DoS and Probe
attacks (traffic-shaped) remain detectable; R2L and U2R (content-shaped) will
largely read as Normal. This is a fundamental limit of running a KDD'99-era
model on live packets, not a bug.

Requirements
------------
- scapy:           pip install scapy   (also in requirements-dev.txt)
- Windows:         install Npcap (https://npcap.com) and run terminal as Admin.
- Linux/macOS:     run with sudo (raw socket capture needs privileges).

Usage
-----
sudo python live_capture.py                       # default iface, all IP traffic
python live_capture.py --iface "Wi-Fi"            # pick interface (Windows name)
python live_capture.py --filter "tcp port 80"     # BPF filter
python live_capture.py --flow-timeout 2           # finalize idle flows after 2s

List interfaces:
    python -c "from scapy.all import get_if_list; print(get_if_list())"
"""

import os
import time
import argparse
from collections import deque

from utils import FEATURE_NAMES, logger

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

STREAM_PATH = os.path.join("data", "stream.csv")

# Common destination port -> NSL-KDD 'service' name (best-effort subset).
PORT_SERVICE = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    37: "time", 43: "whois", 53: "domain", 79: "finger", 80: "http",
    110: "pop_3", 111: "sunrpc", 113: "auth", 119: "nntp", 123: "ntp_u",
    135: "loc_srv", 139: "netbios_ssn", 143: "imap4", 443: "http_443",
    445: "microsoft_ds", 514: "shell", 515: "printer", 993: "imap4",
    995: "pop_3", 3389: "rdp", 8080: "http",
}

# NSL-KDD time-based stats use a 2-second window; host-based stats use the
# trailing 100 connections.
TIME_WINDOW = 2.0
HOST_WINDOW = 100


def port_to_service(port, proto):
    if proto == "icmp":
        return "eco_i"
    return PORT_SERVICE.get(port, "private" if port and port < 1024 else "other")


class Flow:
    """One connection's accumulating state."""

    def __init__(self, key, proto, src, dst, sport, dport, now):
        self.key = key
        self.proto = proto
        self.src, self.dst = src, dst
        self.sport, self.dport = sport, dport
        self.first_ts = self.last_ts = now
        self.src_bytes = 0
        self.dst_bytes = 0
        self.wrong_fragment = 0
        self.urgent = 0
        # TCP handshake/teardown observations for flag derivation.
        self.syn = self.synack = self.fin = self.rst = False
        self.established = False

    def update(self, pkt, now, payload_len, from_src):
        self.last_ts = now
        if from_src:
            self.src_bytes += payload_len
        else:
            self.dst_bytes += payload_len
        ip = pkt[IP]
        if ip.frag != 0 or (ip.flags & 0x1):  # MF set or non-zero offset
            self.wrong_fragment += 1
        if TCP in pkt:
            t = pkt[TCP]
            f = int(t.flags)
            if f & 0x20:  # URG
                self.urgent += 1
            if from_src and (f & 0x02) and not (f & 0x10):  # SYN, not ACK
                self.syn = True
            if (not from_src) and (f & 0x12) == 0x12:        # SYN+ACK
                self.synack = True
                self.established = True
            if f & 0x01:  # FIN
                self.fin = True
            if f & 0x04:  # RST
                self.rst = True

    def flag(self):
        """Approximate NSL-KDD TCP 'flag' from observed handshake."""
        if self.proto != "tcp":
            return "SF"  # udp/icmp: treat as complete
        if self.rst:
            return "RSTO" if self.established else "REJ"
        if self.syn and not self.synack:
            return "S0"          # SYN sent, never answered
        if self.established and self.fin:
            return "SF"          # full open + close
        if self.established:
            return "SF"          # open, data exchanged
        return "OTH"

    def service(self):
        return port_to_service(self.dport, self.proto)

    def duration(self):
        return int(self.last_ts - self.first_ts)


class Tracker:
    """Assembles packets into flows and emits NSL-KDD feature rows."""

    def __init__(self, flow_timeout=2.0):
        self.flows = {}                       # key -> Flow
        self.flow_timeout = flow_timeout
        self.history = deque(maxlen=2000)     # finalized (ts, dst, service, flag)

    @staticmethod
    def _key(proto, src, dst, sport, dport):
        # Direction-insensitive flow key (so replies match the same flow).
        a, b = (src, sport), (dst, dport)
        lo, hi = sorted([a, b])
        return (proto, lo, hi)

    def on_packet(self, pkt):
        if IP not in pkt:
            return
        now = time.time()
        ip = pkt[IP]
        if TCP in pkt:
            proto, l4 = "tcp", pkt[TCP]
            sport, dport = int(l4.sport), int(l4.dport)
            payload = len(l4.payload)
        elif UDP in pkt:
            proto, l4 = "udp", pkt[UDP]
            sport, dport = int(l4.sport), int(l4.dport)
            payload = len(l4.payload)
        elif ICMP in pkt:
            proto = "icmp"
            sport = dport = 0
            payload = len(pkt[ICMP].payload)
        else:
            return

        key = self._key(proto, ip.src, ip.dst, sport, dport)
        flow = self.flows.get(key)
        if flow is None:
            # First packet defines client->server direction.
            flow = Flow(key, proto, ip.src, ip.dst, sport, dport, now)
            self.flows[key] = flow
        from_src = (ip.src == flow.src and sport == flow.sport)
        flow.update(pkt, now, payload, from_src)

        self.flush(now)

    def flush(self, now):
        """Finalize and emit flows idle longer than flow_timeout."""
        done = [k for k, f in self.flows.items()
                if now - f.last_ts >= self.flow_timeout]
        for k in done:
            self._emit(self.flows.pop(k), now)

    def _window_conns(self, now):
        return [h for h in self.history if now - h[0] <= TIME_WINDOW]

    def _emit(self, flow, now):
        service = flow.service()
        flag = flow.flag()

        # --- time-window (2s) stats BEFORE adding this connection -----------
        win = self._window_conns(now)
        same_host = [h for h in win if h[1] == flow.dst]
        same_srv = [h for h in win if h[2] == service]
        count = len(same_host)
        srv_count = len(same_srv)

        def rate(subset, pred):
            return round(sum(pred(h) for h in subset) / len(subset), 2) \
                if subset else 0.0

        serror = lambda h: h[3] in ("S0", "S1", "S2", "S3")
        rerror = lambda h: h[3] == "REJ"

        serror_rate = rate(same_host, serror)
        srv_serror_rate = rate(same_srv, serror)
        rerror_rate = rate(same_host, rerror)
        srv_rerror_rate = rate(same_srv, rerror)
        same_srv_rate = rate(same_host, lambda h: h[2] == service)
        diff_srv_rate = rate(same_host, lambda h: h[2] != service)
        srv_diff_host_rate = rate(same_srv, lambda h: h[1] != flow.dst)

        # --- host-window (last 100 conns) stats -----------------------------
        recent = list(self.history)[-HOST_WINDOW:]
        rec_host = [h for h in recent if h[1] == flow.dst]
        rec_srv = [h for h in recent if h[2] == service]
        dst_host_count = len(rec_host)
        dst_host_srv_count = len(rec_srv)
        dst_host_same_srv_rate = rate(rec_host, lambda h: h[2] == service)
        dst_host_diff_srv_rate = rate(rec_host, lambda h: h[2] != service)
        dst_host_same_src_port_rate = rate(rec_srv, lambda h: True)  # approx
        dst_host_srv_diff_host_rate = rate(rec_srv, lambda h: h[1] != flow.dst)
        dst_host_serror_rate = rate(rec_host, serror)
        dst_host_srv_serror_rate = rate(rec_srv, serror)
        dst_host_rerror_rate = rate(rec_host, rerror)
        dst_host_srv_rerror_rate = rate(rec_srv, rerror)

        land = 1 if (flow.src == flow.dst and flow.sport == flow.dport) else 0

        # --- assemble full 41-feature row (content features zero-filled) ----
        feats = {name: 0 for name in FEATURE_NAMES}
        feats.update({
            "duration": flow.duration(),
            "protocol_type": flow.proto,
            "service": service,
            "flag": flag,
            "src_bytes": flow.src_bytes,
            "dst_bytes": flow.dst_bytes,
            "land": land,
            "wrong_fragment": flow.wrong_fragment,
            "urgent": flow.urgent,
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror_rate,
            "srv_serror_rate": srv_serror_rate,
            "rerror_rate": rerror_rate,
            "srv_rerror_rate": srv_rerror_rate,
            "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate,
            "srv_diff_host_rate": srv_diff_host_rate,
            "dst_host_count": dst_host_count,
            "dst_host_srv_count": dst_host_srv_count,
            "dst_host_same_srv_rate": dst_host_same_srv_rate,
            "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
            "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
            "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
            "dst_host_serror_rate": dst_host_serror_rate,
            "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
            "dst_host_rerror_rate": dst_host_rerror_rate,
            "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate,
        })

        # Record this connection in history for future windows.
        self.history.append((now, flow.dst, service, flag))

        line = ",".join(str(feats[name]) for name in FEATURE_NAMES)
        _append_stream(line)
        logger.info("flow %s:%s->%s:%s %s/%s bytes=%d/%d flag=%s",
                    flow.src, flow.sport, flow.dst, flow.dport,
                    flow.proto, service, flow.src_bytes, flow.dst_bytes, flag)


def _ensure_stream_header():
    os.makedirs(os.path.dirname(STREAM_PATH), exist_ok=True)
    if not os.path.exists(STREAM_PATH):
        with open(STREAM_PATH, "w", encoding="utf-8") as f:
            f.write(",".join(FEATURE_NAMES) + "\n")


def _append_stream(line):
    with open(STREAM_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Live packet capture -> stream.csv")
    ap.add_argument("--iface", default=None,
                    help="Interface to sniff (default: scapy's default)")
    ap.add_argument("--filter", default="ip",
                    help="BPF filter (default 'ip')")
    ap.add_argument("--flow-timeout", type=float, default=2.0,
                    help="Seconds of inactivity before a flow is finalized")
    args = ap.parse_args()

    if not HAS_SCAPY:
        raise SystemExit(
            "scapy not installed. Run: pip install scapy\n"
            "Windows also needs Npcap (https://npcap.com) + Admin terminal.")

    _ensure_stream_header()
    tracker = Tracker(flow_timeout=args.flow_timeout)
    logger.info("Sniffing on %s (filter=%r) -> %s. Ctrl+C to stop.",
                args.iface or "default iface", args.filter, STREAM_PATH)
    logger.info("Reminder: content features are zero-filled (see module docstring).")

    try:
        sniff(iface=args.iface, filter=args.filter, store=False,
              prn=tracker.on_packet)
    except PermissionError:
        raise SystemExit("Permission denied — run as Admin (Windows) / sudo (Unix).")
    except KeyboardInterrupt:
        # Flush whatever is still open.
        tracker.flush(time.time() + tracker.flow_timeout + 1)
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
