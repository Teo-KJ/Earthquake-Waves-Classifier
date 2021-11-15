import datetime

class ColDataTypes:

    initial_dtype_dict = {
        "network_code": str, 
        "receiver_code": str,
        "receiver_type": str, 
        "receiver_latitude": float,
        "receiver_longitude": float, 
        "receiver_elevation_m": float,
        "p_arrival_sample": float, 
        "p_status": str,
        "p_weight": float, 
        "p_travel_sec": float,
        "s_arrival_sample": float, 
        "s_status": str,
        "s_weight": float, 
        "source_id": str,
        "source_origin_uncertainty_sec": str,
        "source_latitude": float, 
        "source_longitude": float,
        "source_error_sec": str, 
        "source_gap_deg": str,
        "source_horizontal_uncertainty_km": str, 
        "source_depth_km": str,
        "source_depth_uncertainty_km": str, 
        "source_magnitude": float,
        "source_magnitude_type": str, 
        "source_magnitude_author": str,
        "source_mechanism_strike_dip_rake": str, 
        "source_distance_deg": float,
        "source_distance_km": float, 
        "back_azimuth_deg": float,
        "snr_db": str, 
        "coda_end_sample": str,
        "trace_category": str,
        "trace_name": str, 
        # "source_origin_time": datetime.datetime, 
        # "trace_start_time": datetime.datetime, 
    }

    dtype_dict = {

    }

    date_cols = [
        'source_origin_time', 
        'trace_start_time'
    ]

    def __init__(self) -> None:
        pass

    def get_initial_dtype_dict(self):
        return self.initial_dtype_dict

    def get_dtype_dict(self):
        return self.dtype_dict

    def get_date_cols(self):
        return self.date_cols

    def get_col_type(self, col_name):
        if self.dtype_dict:
            return self.dtype_dict[col_name]
        else:
            return self.initial_dtype_dict[col_name]