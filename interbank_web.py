#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal web server for interbank simulate, multiple and dashboard modes."""

from concurrent.futures import ProcessPoolExecutor
import math
import os

import numpy as np
import interbank
from flask import Flask, jsonify, render_template, request


def _coerce_value(current_value, new_value):
    if isinstance(current_value, bool):
        if isinstance(new_value, str):
            return new_value.lower() in ("y", "yes", "t", "true", "on", "1")
        return bool(new_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(new_value)
    if isinstance(current_value, float):
        return float(new_value)
    return new_value


def _sanitize_json_value(value):
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _sanitize_series(values):
    return [_sanitize_json_value(value) for value in values]


def _build_aliases(series):
    aliases = {
        "interest_rate": "ir",
        "bankruptcy": "bankruptcies",
        "B": "bad_debt",
    }
    for alias, source in aliases.items():
        if alias not in series and source in series:
            series[alias] = list(series[source])


def _parse_config(payload):
    model = interbank.Model()
    config_values = {}
    for key, value in payload.items():
        if hasattr(model.config, key):
            current_value = getattr(model.config, key)
            config_values[key] = _coerce_value(current_value, value)
    return config_values


def _simulate(payload):
    model = interbank.Model()
    config_values = _parse_config(payload)

    if config_values:
        model.configure(**config_values)

    model.log.interactive = False
    result = model.run()

    series = {}
    for column in result.columns:
        series[column] = _sanitize_series(result[column].tolist())
    _build_aliases(series)

    meta = {
        "N": model.config.N,
        "T": model.config.T,
        "seed": model.config.seed,
        "rows": len(result),
    }

    return {
        "meta": meta,
        "time": list(range(len(result))),
        "series": series,
    }


SUM_METRICS = {
    "bankruptcies",
    "bad_debt",
    "rationing",
    "num_loans",
}
LAST_METRICS = {
    "num_banks",
}
DEFAULT_MULTIPLE_METRICS = ["bankruptcies", "equity", "ir"]


def _normalize_metric_name(metric):
    aliases = {
        "bankruptcy": "bankruptcies",
        "interest_rate": "ir",
        "B": "bad_debt",
    }
    return aliases.get(metric, metric)


def _extract_metric(model, metric_name):
    values = getattr(model.stats, metric_name, None)
    if values is None or len(values) == 0:
        return float("nan")
    array = np.asarray(values, dtype=float)
    if metric_name in LAST_METRICS:
        return float(array[-1])
    if metric_name in SUM_METRICS:
        return float(np.nansum(array))
    tail_start = max(0, len(array) - 100)
    return float(np.nanmean(array[tail_start:]))


def _run_multiple_job(args):
    config_values, multiple_param, multiple_value, seed, metrics = args
    local_config = dict(config_values)
    local_config[multiple_param] = multiple_value
    local_config["seed"] = seed

    model = interbank.Model()
    model.configure(**local_config)
    model.log.interactive = False
    model.run()

    row = {}
    for metric in metrics:
        row[metric] = _extract_metric(model, metric)
    return multiple_value, row


def _build_multiple_values(start, end, step):
    values = []
    current = start
    epsilon = abs(step) * 1e-6
    if step > 0:
        while current <= end + epsilon:
            values.append(round(current, 10))
            current += step
    else:
        while current >= end - epsilon:
            values.append(round(current, 10))
            current += step
    return values


def _multiple(payload):
    multiple_param = payload.get("multiple_param", "p")
    multiple_from = float(payload.get("from", 0.0))
    multiple_to = float(payload.get("to", 1.0))
    multiple_step = float(payload.get("step", 0.1))
    if multiple_step == 0:
        raise ValueError("step must be non-zero")

    config_payload = payload.get("config", {})
    config_values = _parse_config(config_payload)

    model = interbank.Model()
    if not hasattr(model.config, multiple_param):
        raise ValueError(f"Unknown multiple parameter '{multiple_param}'")

    metrics = payload.get("metrics", DEFAULT_MULTIPLE_METRICS)
    metrics = [_normalize_metric_name(metric) for metric in metrics]
    for metric in metrics:
        if not hasattr(model.stats, metric):
            raise ValueError(f"Unknown metric '{metric}'")

    mc_seeds = int(payload.get("mc_seeds", payload.get("mc", 1)))
    mc_seeds = max(mc_seeds, 1)
    workers = int(payload.get("workers", 4))
    workers = max(workers, 1)
    base_seed = int(config_values.get("seed", model.config.seed))

    raw_multiple_values = _build_multiple_values(multiple_from, multiple_to, multiple_step)
    multiple_values = [_coerce_value(getattr(model.config, multiple_param), value) for value in raw_multiple_values]
    jobs = []
    for multiple_value in multiple_values:
        for seed_offset in range(mc_seeds):
            jobs.append((config_values, multiple_param, multiple_value, base_seed + seed_offset, metrics))

    grouped = {value: {metric: [] for metric in metrics} for value in multiple_values}
    max_workers = min(workers, len(jobs)) if jobs else 1

    if jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for multiple_value, row in executor.map(_run_multiple_job, jobs):
                for metric in metrics:
                    grouped[multiple_value][metric].append(row[metric])

    aggregated = {metric: [] for metric in metrics}
    for multiple_value in multiple_values:
        for metric in metrics:
            values = grouped[multiple_value][metric]
            aggregated[metric].append(_sanitize_json_value(float(np.nanmean(values))) if values else None)

    for alias, source in {"bankruptcy": "bankruptcies", "interest_rate": "ir", "B": "bad_debt"}.items():
        if source in aggregated and alias not in aggregated:
            aggregated[alias] = list(aggregated[source])

    return {
        "multiple_param": multiple_param,
        "multiple_values": multiple_values,
        "metrics": aggregated,
        "mc_seeds": mc_seeds,
    }


def create_app(web_mode="dashboard"):
    template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    app = Flask(__name__, template_folder=template_folder)
    app.config["WEB_MODE"] = web_mode

    @app.route("/")
    def index():
        if app.config["WEB_MODE"] == "simulate":
            return render_template("index_web_simulate.html", web_mode=app.config["WEB_MODE"])
        if app.config["WEB_MODE"] == "multiple":
            return render_template("index_web_multiple.html", web_mode=app.config["WEB_MODE"])
        if app.config["WEB_MODE"] == "dashboard":
            return render_template("template_simulation.html", web_mode=app.config["WEB_MODE"])
        return jsonify({"error": "invalid web_mode"}), 400

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        if app.config["WEB_MODE"] not in ("simulate", "multiple", "dashboard"):
            return jsonify({"error": f"web_mode '{app.config['WEB_MODE']}' not implemented yet"}), 501
        payload = request.get_json(silent=True) or {}
        try:
            result = _simulate(payload)
            return jsonify(result)
        except Exception as error:  # pragma: no cover - defensive path
            return jsonify({"error": str(error)}), 400

    @app.route("/api/multiple", methods=["POST"])
    def api_multiple():
        if app.config["WEB_MODE"] not in ("multiple", "dashboard"):
            return jsonify({"error": f"web_mode '{app.config['WEB_MODE']}' does not support multiple"}), 501
        payload = request.get_json(silent=True) or {}
        try:
            result = _multiple(payload)
            return jsonify(result)
        except Exception as error:  # pragma: no cover - defensive path
            return jsonify({"error": str(error)}), 400

    return app
