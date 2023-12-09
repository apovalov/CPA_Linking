=from collections import defaultdict

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
    if reward >= 1:
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
    """Thompson sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Choose offer ID by Thompson sampling
    alpha = np.array([offer_actions[offer_id] for offer_id in offers_ids])
    beta = np.array(
        [offer_clicks[offer_id] - offer_actions[offer_id] for offer_id in offers_ids]
    )

    # If there are no clicks for any offer, choose randomly
    if alpha.sum() == 0:
        offer_id = int(np.random.choice(offers_ids))
    else:
        cr = np.random.beta(alpha + 2, beta + 10)
        rpc = np.array(
            [
                offer_rewards[offer_id] / max(offer_clicks[offer_id], 1)
                for offer_id in offers_ids
            ]
        )
        offer_id = offers_ids[np.argmax(cr * rpc)]

    # Update statistics
    pending_clicks[click_id] = offer_id
    offer_clicks[offer_id] += 1

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()
