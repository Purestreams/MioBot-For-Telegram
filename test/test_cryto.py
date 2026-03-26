import asyncio

import app.cryto as cryto


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_get_price_filters_tokens(monkeypatch):
    async def fake_aget(url, params=None, retries=3, backoff=0.5):
        return _Resp([
            {"token": "SOL", "usdPrice": "123.456"},
            {"token": "BTC", "usdPrice": "99999.9"},
        ])

    monkeypatch.setattr(cryto, "_aget", fake_aget)
    result = asyncio.run(cryto.get_Price(["SOL"]))
    assert result == {"SOL": 123.456}


def test_get_allez_apr_formats_fields(monkeypatch):
    async def fake_aget(url, params=None, retries=3, backoff=0.5):
        return _Resp({
            "apy24h": 0.12,
            "apy7d": 0.1,
            "apy30d": 0.08,
            "apy90d": 0.09,
            "tokensInvestedUsd": 5_500_000,
        })

    monkeypatch.setattr(cryto, "_aget", fake_aget)
    result = asyncio.run(cryto.get_Allez_APR())
    assert result["name"] == "Allez SOL"
    assert result["APR_24H"] == "12.0%"
    assert result["Total_Supply"] == "5.5M"


def test_get_price_coinbase_continues_on_single_token_failure(monkeypatch):
    async def fake_aget(url, params=None, retries=3, backoff=0.5):
        if "BTC-USD" in url:
            raise RuntimeError("boom")
        return _Resp({"data": {"amount": "1.2345"}})

    monkeypatch.setattr(cryto, "_aget", fake_aget)
    result = asyncio.run(cryto.get_Price_Coinbase(["BTC", "SOL"]))
    assert "BTC" not in result
    assert result["SOL"] == 1.234
