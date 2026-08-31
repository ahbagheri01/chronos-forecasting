"""Versioned loaders for independently sourced official benchmark datasets.

Each loader downloads a fixed query only when its JSON snapshot is absent.  Later
runs consume that snapshot and its SHA-256 sidecar, which prevents silent data
revisions from changing an experiment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from pmds.datasets import make_tasks, select_representative_items
from pmds.schemas import DatasetSpec, ForecastTask


LOGGER = logging.getLogger("pmds.compare")
USER_AGENT = "chronos-forecasting-pmds/1.0 (research benchmark)"


def _get_json(url: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    query = urlencode(params or {}, doseq=True)
    request_url = f"{url}?{query}" if query else url
    request = Request(request_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:  # nosec B310: fixed official endpoints are configured below
        return json.loads(response.read().decode("utf-8"))


def _required_env(name: str, dataset: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{dataset} needs environment variable {name} the first time its fixed snapshot is downloaded"
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    digest = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def _read_snapshot(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Snapshot checksum mismatch: {path}")
    payload = json.loads(content)
    if not isinstance(payload.get("series"), dict) or not payload["series"]:
        raise ValueError(f"Snapshot contains no series: {path}")
    return payload


def _normalize_series(
    observations: list[tuple[Any, Any]],
    frequency: str,
    start: str,
    end: str,
) -> dict[str, list[Any]]:
    if not observations:
        return {"timestamps": [], "values": []}
    timestamps = pd.to_datetime([item[0] for item in observations], utc=True, errors="coerce")
    values = pd.to_numeric([item[1] for item in observations], errors="coerce")
    valid = ~pd.isna(timestamps)
    series = pd.Series(np.asarray(values)[valid], index=pd.DatetimeIndex(timestamps[valid]))
    series.index = series.index.tz_convert(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    expected = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=frequency)
    series = series.reindex(expected)
    normalized_values = [None if not np.isfinite(value) else float(value) for value in series.to_numpy(dtype=float)]
    return {
        "timestamps": [timestamp.isoformat() for timestamp in series.index],
        "values": normalized_values,
    }


def _download_eia_930(spec: DatasetSpec) -> dict[str, Any]:
    params = spec.source_params
    api_key = _required_env(str(params.get("api_key_env", "EIA_API_KEY")), spec.name)
    endpoint = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    start, end = str(params["start"]), str(params["end"])
    authorities = list(map(str, params["authorities"]))
    result: dict[str, Any] = {}
    for authority in authorities:
        observations: list[tuple[Any, Any]] = []
        offset = 0
        while True:
            response = _get_json(
                endpoint,
                {
                    "api_key": api_key,
                    "frequency": "hourly",
                    "data[0]": "value",
                    "facets[respondent][]": authority,
                    "facets[type][]": "D",
                    "start": start,
                    "end": end,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                    "offset": offset,
                    "length": 5000,
                },
            )
            body = response.get("response", {})
            rows = body.get("data", [])
            observations.extend((row.get("period"), row.get("value")) for row in rows)
            offset += len(rows)
            total = int(body.get("total", offset))
            if not rows or offset >= total:
                break
        normalized = _normalize_series(observations, "h", start, end)
        if not normalized["values"]:
            raise RuntimeError(f"EIA returned no hourly demand observations for {authority}")
        result[authority] = normalized
    return {
        "dataset": "eia_930_hourly_demand",
        "downloaded_at": _utc_now(),
        "source": endpoint,
        "query": {"authorities": authorities, "type": "D", "start": start, "end": end},
        "series": result,
    }


def _download_usgs_streamflow(spec: DatasetSpec) -> dict[str, Any]:
    params = spec.source_params
    endpoint = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
    start, end = str(params["start"]), str(params["end"])
    gauges = list(map(str, params["gauges"]))
    parameter_code = str(params.get("parameter_code", "00060"))
    statistic_id = str(params.get("statistic_id", "00003"))
    result: dict[str, Any] = {}
    for gauge in gauges:
        next_url: str | None = endpoint
        query: Mapping[str, Any] | None = {
            "f": "json",
            "limit": 10000,
            "monitoring_location_id": f"USGS-{gauge}",
            "parameter_code": parameter_code,
            "statistic_id": statistic_id,
            "datetime": f"{start}/{end}",
        }
        observations: list[tuple[Any, Any]] = []
        while next_url is not None:
            response = _get_json(next_url, query)
            query = None
            observations.extend(
                (feature.get("properties", {}).get("time"), feature.get("properties", {}).get("value"))
                for feature in response.get("features", [])
            )
            next_url = next(
                (link.get("href") for link in response.get("links", []) if link.get("rel") == "next"),
                None,
            )
        normalized = _normalize_series(observations, "D", start, end)
        if not normalized["values"]:
            raise RuntimeError(f"USGS returned no daily streamflow observations for gauge {gauge}")
        result[gauge] = normalized
    return {
        "dataset": "usgs_daily_streamflow",
        "downloaded_at": _utc_now(),
        "source": endpoint,
        "query": {
            "gauges": gauges,
            "parameter_code": parameter_code,
            "statistic_id": statistic_id,
            "start": start,
            "end": end,
        },
        "series": result,
    }


def _download_fred_md(spec: DatasetSpec) -> dict[str, Any]:
    params = spec.source_params
    api_key = _required_env(str(params.get("api_key_env", "FRED_API_KEY")), spec.name)
    endpoint = "https://api.stlouisfed.org/fred/series/observations"
    start, end = str(params["start"]), str(params["end"])
    vintage_date = str(params["vintage_date"])
    series_ids = list(map(str, params["series_ids"]))
    result: dict[str, Any] = {}
    for series_id in series_ids:
        response = _get_json(
            endpoint,
            {
                "api_key": api_key,
                "file_type": "json",
                "series_id": series_id,
                "observation_start": start,
                "observation_end": end,
                "frequency": "m",
                "aggregation_method": "avg",
                "realtime_start": vintage_date,
                "realtime_end": vintage_date,
            },
        )
        observations = [(row.get("date"), row.get("value")) for row in response.get("observations", [])]
        normalized = _normalize_series(observations, "MS", start, end)
        if not normalized["values"]:
            raise RuntimeError(f"FRED returned no vintage observations for {series_id}")
        result[series_id] = normalized
    return {
        "dataset": "fred_md_fixed_vintage",
        "downloaded_at": _utc_now(),
        "source": endpoint,
        "query": {"series_ids": series_ids, "start": start, "end": end, "vintage_date": vintage_date},
        "series": result,
    }


DOWNLOADERS = {
    "eia_930": _download_eia_930,
    "usgs_streamflow": _download_usgs_streamflow,
    "fred_md": _download_fred_md,
}


def load_official_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    if not spec.source or spec.source not in DOWNLOADERS:
        raise ValueError(f"Unsupported official source '{spec.source}' for {spec.name}")
    if not spec.snapshot_path:
        raise ValueError(f"Official dataset {spec.name} requires snapshot_path")
    snapshot_path = Path(spec.snapshot_path).expanduser()
    if not snapshot_path.exists():
        LOGGER.info("Downloading fixed official-data snapshot | dataset=%s path=%s", spec.name, snapshot_path)
        payload = DOWNLOADERS[spec.source](spec)
        payload["benchmark"] = {
            "frequency": spec.fallback_frequency,
            "prediction_length": spec.prediction_length,
            "seasonality": spec.seasonality,
            "num_origins": spec.num_origins,
            "origin_stride": spec.origin_stride or spec.prediction_length,
            "origin_rule": "latest complete target then fixed backward strides",
            "max_series": spec.max_series,
            "selection_strategy": spec.selection_strategy,
        }
        _write_snapshot(snapshot_path, payload)
    snapshot = _read_snapshot(snapshot_path)
    item_ids = select_representative_items(
        sorted(snapshot["series"]), spec.max_series, spec.selection_strategy, spec.selection_seed
    )
    tasks: list[ForecastTask] = []
    for item_id in item_ids:
        item = snapshot["series"][item_id]
        tasks.extend(make_tasks(spec, item_id, item["values"], item["timestamps"]))
    return tasks
