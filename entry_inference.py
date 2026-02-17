# entry_inference.py

import json
import yaml
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from datetime import time


class EntryInferenceEngine:
    def __init__(self, model_dir: str):
        # ---------- Load configs ----------
        with open(f"{model_dir}/feature_list.json") as f:
            self.feature_list = json.load(f)

        with open(f"{model_dir}/thresholds.json") as f:
            self.thresholds = json.load(f)

        with open(f"{model_dir}/entry_config.yaml") as f:
            self.config = yaml.safe_load(f)

        # ---------- Load models ----------
        self.lgb_long = lgb.Booster(model_file=f"{model_dir}/lgb_long.txt")
        self.lgb_short = lgb.Booster(model_file=f"{model_dir}/lgb_short.txt")

        self.xgb_long = xgb.Booster()
        self.xgb_long.load_model(f"{model_dir}/xgb_long.json")

        self.xgb_short = xgb.Booster()
        self.xgb_short.load_model(f"{model_dir}/xgb_short.json")

        # ---------- Ensemble weights ----------
        self.w_lgb = self.config["ensemble"]["lgb_weight"]
        self.w_xgb = self.config["ensemble"]["xgb_weight"]

        # ---------- Market hours ----------
        self.market_open = self._parse_time(self.config["market"]["open"])
        self.market_close = self._parse_time(self.config["market"]["close"])

        self.allow_long = self.config["entry"]["allow_long"]
        self.allow_short = self.config["entry"]["allow_short"]

    # --------------------------------------------------
    def _parse_time(self, t: str):
        hh, mm = map(int, t.split(":"))
        return time(hh, mm)

    def _market_open_allowed(self, ts):
        t = ts.time()
        return self.market_open <= t <= self.market_close

    # --------------------------------------------------
    def _build_feature_vector(self, feature_dict: dict):
        """
        Enforces exact feature order
        """
        return np.array(
            [feature_dict[f] for f in self.feature_list],
            dtype=np.float32
        ).reshape(1, -1)

    # --------------------------------------------------
    def predict(self, feature_dict: dict, symbol: str, timestamp):
        """
        Returns: 'LONG', 'SHORT', or None
        """

        # ---------- Session filter ----------
        if not self._market_open_allowed(timestamp):
            return None

        # ---------- Build feature vector ----------
        X = self._build_feature_vector(feature_dict)

        # ---------- LightGBM ----------
        lgb_long_p = self.lgb_long.predict(X)[0]
        lgb_short_p = self.lgb_short.predict(X)[0]

        # ---------- XGBoost ----------
        dmat = xgb.DMatrix(X, feature_names=self.feature_list)
        xgb_long_p = self.xgb_long.predict(dmat)[0]
        xgb_short_p = self.xgb_short.predict(dmat)[0]

        # ---------- Ensemble ----------
        long_score = self.w_lgb * lgb_long_p + self.w_xgb * xgb_long_p
        short_score = self.w_lgb * lgb_short_p + self.w_xgb * xgb_short_p

        # Use ensemble threshold (average of lgb and xgb thresholds)
        long_th = self.w_lgb * self.thresholds["lgb_long"] + self.w_xgb * self.thresholds["xgb_long"]
        short_th = self.w_lgb * self.thresholds["lgb_short"] + self.w_xgb * self.thresholds["xgb_short"]

        # ---------- Decision logic ----------
        if self.allow_long and long_score > long_th and long_score > short_score:
            return "LONG"

        if self.allow_short and short_score > short_th and short_score > long_score:
            return "SHORT"

        return None
