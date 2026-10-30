"""
Network Intrusion Detection - Feature Engineering Pipeline
Extracts domain-grounded network traffic features:
- Port categorization (Well-Known, Registered, Dynamic) and service indicators
- Traffic flow directionality and packet/byte asymmetries
- TCP control flag dynamics and intensity ratios
- Window size mechanics and tool fingerprints
- Duration-floored logarithmic rate metrics
- Domain attack signatures (SYN flood, slowloris, UDP scanning)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DROP_COLS = ['id', 'id.orig_p', 'fwd_URG_flag_count']

def categorize_port(p):
    """Categorizes port into IANA standard tiers."""
    try:
        p = float(p)
        if np.isnan(p):
            return "missing"
        p = int(p)
        if p < 1024:
            return "well_known"
        elif p <= 49151:
            return "registered"
        else:
            return "dynamic"
    except Exception:
        return "missing"

class NetworkFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature extraction pipeline producing 133 domain-specific features.
    Guarantees no NaN, Inf, or negative rates with zero division protection.
    """
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.port_cat_map_ = {'well_known': 0, 'registered': 1, 'dynamic': 2, 'missing': -1}
        self.proto_map_ = {}
        self.service_map_ = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        df = X.copy()
        if 'proto' in df.columns:
            protos = sorted(df['proto'].dropna().astype(str).unique())
            self.proto_map_ = {p: i for i, p in enumerate(protos)}
        if 'service' in df.columns:
            services = sorted(df['service'].dropna().astype(str).unique())
            self.service_map_ = {s: i for i, s in enumerate(services)}
        return self

    def transform(self, X):
        df = X.copy()
        eps = self.eps

        # 1. Drop identifiers
        drop_list = [c for c in DROP_COLS if c in df.columns]
        if 'Target' in df.columns:
            drop_list.append('Target')
        df = df.drop(columns=drop_list, errors='ignore')

        # 2. Port Semantics & Service Indicators
        if 'id.resp_p' in df.columns:
            port_cats = df['id.resp_p'].apply(categorize_port)
            df['port_tier'] = port_cats.map(self.port_cat_map_).fillna(-1).astype(int)
            
            # Dedicated indicators for core services
            df['is_port_ftp_21'] = (df['id.resp_p'] == 21).astype(float)
            df['is_port_ssh_22'] = (df['id.resp_p'] == 22).astype(float)
            df['is_port_dns_53'] = (df['id.resp_p'] == 53).astype(float)
            df['is_port_http_80'] = (df['id.resp_p'] == 80).astype(float)
            df['is_port_https_443'] = (df['id.resp_p'] == 443).astype(float)
            df['is_port_mqtt_1883'] = (df['id.resp_p'] == 1883).astype(float)
            df['is_port_mdns_5353'] = (df['id.resp_p'] == 5353).astype(float)

        # 3. Categorical Encodings
        if 'proto' in df.columns:
            df['proto_enc'] = df['proto'].astype(str).map(self.proto_map_).fillna(-1).astype(int)
            df = df.drop(columns=['proto'])
        if 'service' in df.columns:
            df['service_enc'] = df['service'].astype(str).map(self.service_map_).fillna(-1).astype(int)
            df = df.drop(columns=['service'])

        # 4. Traffic Asymmetry & Directionality
        fwd_p = df['fwd_pkts_tot'].clip(lower=0) if 'fwd_pkts_tot' in df.columns else 0
        bwd_p = df['bwd_pkts_tot'].clip(lower=0) if 'bwd_pkts_tot' in df.columns else 0
        df['pkts_ratio'] = (fwd_p + 1.0) / (bwd_p + 1.0)
        df['pkt_asymmetry'] = (fwd_p - bwd_p) / (fwd_p + bwd_p + eps)

        fwd_b = df['fwd_subflow_bytes'].clip(lower=0) if 'fwd_subflow_bytes' in df.columns else 0
        bwd_b = df['bwd_subflow_bytes'].clip(lower=0) if 'bwd_subflow_bytes' in df.columns else 0
        df['bytes_ratio'] = (fwd_b + 1.0) / (bwd_b + 1.0)
        df['byte_asymmetry'] = (fwd_b - bwd_b) / (fwd_b + bwd_b + eps)

        # Directional Density
        if 'fwd_pkts_payload.tot' in df.columns:
            df['fwd_bytes_per_pkt'] = df['fwd_pkts_payload.tot'].clip(lower=0) / (fwd_p + eps)
        if 'bwd_pkts_payload.tot' in df.columns:
            df['bwd_bytes_per_pkt'] = df['bwd_pkts_payload.tot'].clip(lower=0) / (bwd_p + eps)
        if 'flow_pkts_payload.tot' in df.columns:
            tot_p = fwd_p + bwd_p
            df['flow_bytes_per_pkt'] = df['flow_pkts_payload.tot'].clip(lower=0) / (tot_p + eps)

        # Directional Flags
        df['is_unidirectional'] = (bwd_p == 0).astype(float)
        if 'flow_duration' in df.columns:
            df['is_zero_duration'] = (df['flow_duration'] <= 0).astype(float)
        if 'bwd_pkts_payload.tot' in df.columns:
            df['bwd_payload_is_zero'] = (df['bwd_pkts_payload.tot'] == 0).astype(float)

        # 5. TCP Flag Dynamics
        flag_cols = ['flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'flow_ACK_flag_count']
        if all(c in df.columns for c in flag_cols):
            df['syn_only'] = ((df['flow_SYN_flag_count'] > 0) & (df['flow_ACK_flag_count'] == 0)).astype(float)
            df['fin_only'] = ((df['flow_FIN_flag_count'] > 0) & (df['flow_SYN_flag_count'] == 0)).astype(float)
            df['total_flags'] = df['flow_FIN_flag_count'] + df['flow_SYN_flag_count'] + df['flow_RST_flag_count'] + df['flow_ACK_flag_count']
            df['rst_syn_ratio'] = (df['flow_RST_flag_count'] + 1.0) / (df['flow_SYN_flag_count'] + 1.0)

        # 6. Window Size Mechanics & Fingerprints
        if 'fwd_init_window_size' in df.columns and 'bwd_init_window_size' in df.columns:
            df['window_ratio'] = (df['fwd_init_window_size'].clip(lower=0) + 1.0) / (df['bwd_init_window_size'].clip(lower=0) + 1.0)
            df['win_fingerprint_64'] = (df['fwd_init_window_size'] == 64).astype(float)
            df['win_fingerprint_0'] = (df['fwd_init_window_size'] == 0).astype(float)
            df['win_fingerprint_64240'] = (df['fwd_init_window_size'] == 64240).astype(float)
            df['win_fingerprint_65535'] = (df['fwd_init_window_size'] == 65535).astype(float)

        # 7. Domain Attack Signatures
        if 'fwd_init_window_size' in df.columns and 'fwd_pkts_payload.tot' in df.columns:
            df['sig_hping_syn_flood'] = ((df['fwd_init_window_size'] == 64) & (df['fwd_pkts_payload.tot'] == 120)).astype(float)
        if 'id.resp_p' in df.columns and 'bwd_pkts_payload.tot' in df.columns:
            df['sig_slowloris_dos'] = ((df['id.resp_p'] == 80) & (df['bwd_pkts_payload.tot'] == 0) & (fwd_p > 5)).astype(float)
            df['sig_udp_scan'] = ((df['id.resp_p'] != 53) & (df['id.resp_p'] != 5353) & (bwd_p == 0)).astype(float)

        # 8. Robust Rate Metrics (Logarithmic with Duration Floor)
        if 'flow_duration' in df.columns:
            dur_floor = df['flow_duration'].clip(lower=eps)
            df['log_duration'] = np.log1p(dur_floor)
            if 'flow_pkts_tot' in df.columns:
                df['log_pkts_rate'] = np.log1p(df['flow_pkts_tot'] / dur_floor)
            if 'payload_bytes_per_second' in df.columns:
                df['log_payload_rate'] = np.log1p(df['payload_bytes_per_second'].clip(lower=0))

        # Sanitize any inf or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        self.feature_names_ = list(df.columns)
        return df
