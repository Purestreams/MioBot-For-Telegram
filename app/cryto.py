import asyncio
import random
import httpx

# API URLs
url_Price = "https://api.kamino.finance/prices"
PRICE_PARAMS = {"env": "mainnet-beta", "source": "scope"}

url_Allez_USDC = "https://api.kamino.finance/kvaults/A1USdzqDHmw5oz97AkqAGLxEQZfFjASZFuy4T6Qdvnpo/metrics"
url_Allez_SOL = "https://api.kamino.finance/kvaults/A1so1bPD3W1TfeFwboDh8yfAAVaVtcdAYBYCjhg2mJQ/metrics"

# Session-like defaults
DEFAULT_HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "connection": "keep-alive",
}

STATUS_FORCELIST = {429, 500, 502, 503, 504}

_aclient: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _aclient
    if _aclient is None:
        _aclient = httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=httpx.Timeout(10.0))
    return _aclient


async def _aclose_client():
    global _aclient
    if _aclient is not None:
        await _aclient.aclose()
        _aclient = None


async def _aget(url: str, params: dict | None = None, retries: int = 3, backoff: float = 0.5) -> httpx.Response:
    client = await _get_client()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await client.get(url, params=params)
            # Raise for HTTP errors
            resp.raise_for_status()
            # Optionally gate retries based on status (already raised above)
            return resp
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            if code not in STATUS_FORCELIST or attempt == retries - 1:
                raise
        except httpx.TransportError as e:
            last_err = e
            if attempt == retries - 1:
                raise
        # Exponential backoff with jitter
        await asyncio.sleep(backoff * (2 ** attempt) + random.uniform(0, 0.25))
    # Should not reach here
    raise last_err if last_err else RuntimeError("Request failed without exception")


async def get_Price(list_tokens: list) -> dict:
    resp = await _aget(url_Price, params=PRICE_PARAMS)
    full_data = resp.json()  # Expected: list of { token, usdPrice, ... }
    output = []
    for item in full_data:
        if isinstance(item, dict) and item.get("token") in list_tokens:
            #print(item)
            output.append(item)

        if len(output) == len(list_tokens):
            break
    
    return {item["token"]: round(float(item["usdPrice"]), 3) for item in output}


async def get_Price_Coinbase(list_tokens: list) -> dict:
    url = "https://api.coinbase.com/v2/prices"
    output = {}
    for token in list_tokens:
        try:
            resp = await _aget(f"{url}/{token}-USD/spot")
            data = resp.json()  # Expected: { data: { base, currency, amount } }
            amount = data.get("data", {}).get("amount")
            if amount is not None:
                output[token] = round(float(amount), 3)
        except Exception as e:
            print(f"Error fetching price for {token}: {e}")
    return output


async def get_Allez_APR() -> dict:
    resp = await _aget(url_Allez_SOL)
    full_data = resp.json()  # Expected: dict

    apr_24h = full_data.get("apy24h")
    apr_7d = full_data.get("apy7d")
    apr_30d = full_data.get("apy30d")
    apr_90d = full_data.get("apy90d")
    total_supply = full_data.get("tokensInvestedUsd")

    def fmt_pct(x):
        return f"{round(float(x) * 100, 2)}%" if x is not None else None

    def fmt_supply(x):
        return f"{round(float(x) / 1e6, 2)}M" if x is not None else None

    return {
        "name": "Allez SOL",
        "APR_24H": fmt_pct(apr_24h),
        "APR_7D": fmt_pct(apr_7d),
        "APR_30D": fmt_pct(apr_30d),
        "APR_90D": fmt_pct(apr_90d),
        "Total_Supply": fmt_supply(total_supply),
    }


async def get_Allez_USDC_APR() -> dict:
    resp = await _aget(url_Allez_USDC)
    full_data = resp.json()  # Expected: dict

    apr_24h = full_data.get("apy24h")
    apr_7d = full_data.get("apy7d")
    apr_30d = full_data.get("apy30d")
    apr_90d = full_data.get("apy90d")
    total_supply = full_data.get("tokensInvestedUsd")

    def fmt_pct(x):
        return f"{round(float(x) * 100, 2)}%" if x is not None else None

    def fmt_supply(x):
        return f"{round(float(x) / 1e6, 2)}M" if x is not None else None

    return {
        "name": "Allez USDC",
        "APR_24H": fmt_pct(apr_24h),
        "APR_7D": fmt_pct(apr_7d),
        "APR_30D": fmt_pct(apr_30d),
        "APR_90D": fmt_pct(apr_90d),
        "Total_Supply": fmt_supply(total_supply),
    }


async def main():
    tokens = ["SOL", "USDC", "BTC", "ETH", "USDT"]
    try:
        prices = await get_Price(tokens)
        print(prices)

        prices_coinbase = await get_Price_Coinbase(tokens)
        print(prices_coinbase)

        allez_apr = await get_Allez_APR()
        print(allez_apr)

        allez_usdc_apr = await get_Allez_USDC_APR()
        print(allez_usdc_apr)
    finally:
        await _aclose_client()


if __name__ == "__main__":
    asyncio.run(main())
