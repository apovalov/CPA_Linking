from collections import defaultdict

import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()

pending_clicks = {}
offer_clicks = defaultdict(int)
offer_actions = defaultdict(int)
offer_rewards = defaultdict(float)


@app.on_event("startup")
def startup() -> None:
    """Initialize statistics"""
    pending_clicks.clear()
    offer_clicks.clear()
    offer_actions.clear()
    offer_rewards.clear()


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Get offer ID
    offer_id = pending_clicks[click_id]
    del pending_clicks[click_id]

    # Non-zero reward means conversion
    if reward > 0:
        offer_rewards[offer_id] += reward
        offer_actions[offer_id] += 1
        is_conversion = True
    else:
        is_conversion = False

    # Response body consists of click ID
    # and accepted click status (True/False)
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward,
    }

    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    clicks = offer_clicks[offer_id]
    conversions = offer_actions[offer_id]
    reward = offer_rewards[offer_id]

    response = {
        "offer_id": offer_id,
        "clicks": offer_clicks[offer_id],
        "conversions": offer_actions[offer_id],
        "reward": offer_rewards[offer_id],
        "cr": conversions / max(clicks, 1),
        "rpc": reward / max(clicks, 1),
    }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Sample random offer"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample random offer ID
    if sum(offer_clicks.values()) < 100:
        offer_id = int(np.random.choice(offers_ids))
        is_greedy = False
    else:
        top_rpc, top_offer = 0, offers_ids[0]
        for offer in offers_ids:
            if offer not in offer_clicks:
                continue
            rpc = offer_rewards[offer] / max(offer_clicks[offer], 1)
            if rpc > top_rpc:
                top_rpc = rpc
                top_offer = offer
        offer_id = top_offer
        is_greedy = True

    # Update statistics
    pending_clicks[click_id] = offer_id
    offer_clicks[offer_id] += 1

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": "greedy" if is_greedy else "random",
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()
