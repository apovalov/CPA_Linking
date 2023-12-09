# Smart-Link: ε-Greedy

It was about an affiliate network. An affiliate (the same as affiliate program/platform/affiliate agency/affiliate network) is an intermediary between advertisers and traffic arbitrageurs.

The former have a product (website, app, web service) and money.
The latter have traffic (users) and channels to attract it.
We help both to find each other.

CPA-platform (cost-per-action), i.e. we pay arbitrageurs for action (for making the desired action) - not for transitions, not for views, but for purchases/subscriptions/registrations, etc., which are set when an advertiser creates an offer on our platform.


## Smart Campaign
A traffic arbitrageur logs into our platform to choose an offerer to which he will pour traffic. Having selected an offer, he starts a campaign and receives a special link. By clicking on this link, users are redirected to the selected offer and then, when the user performs certain actions on the site, the link will store information about the source from which the user came (from which arbitrageur and thanks to which channel of attraction). Accordingly, when a user on the advertiser's site performs the necessary action, we will know which arbitrageur should be credited for this conversion (to whom to pay money).

Choosing the right offerer is not an easy task. You need to know your audience (your traffic) and have a good understanding of which specific users will be paid for. What is the category of the offerer and how suitable it is for the traffic? What geo (geography: country, language) is the banner designed for, and what geo is the traffic? What topics should the user be interested in in order to take the necessary action? And so on.

How to understand all these questions for a beginner arbitrageur, if he already has traffic, but has no experience?

Smart-Link is a machine learning service that allows you to select a wide group of offers and automatically identify those that will convert best (and/or bring the most money) for the given traffic. Accordingly, an arbitrageur directs his traffic not to 1 specific offer, but to several of them at once.

## 1. Random Sampler

Let's start with the simplest model: we'll show random banners for any click that comes in. Our model consists of two stages: candidate selection (those offers that the user can physically go to and subscribe to, depending on their geography, device, source, etc.) and offer selection (what we will implement).

The candidate model is already implemented.

Our ML service receives a click (click ID) and a set of candidate offerers (offer IDs) at once. We want to take a random one of them. This is more than enough for testing integration with the backend.


```
import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/sample/")
def sample(offer_ids: str) -> dict:
    """Sample random offer"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample random offer ID
    offer_id = int(np.random.choice(offers_ids))

    # Prepare response
    response = {
        "offer_id": offer_id,
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()

```


## 2. Greedy Sampler

For the first 100 clicks sent to the service, a random one was selected (for initialization)
For the subsequent ones, the one (among candidate banners) that maximizes RPC was selected. The selection should be made among the candidate banners passed in the offer_ids argument
If no suitable offerer is found, we return the very first one. For example, the following offer_ids came in: [45, 67]. But the statistics for both of them is zero. Then we choose 45

## 3. ε-Greedy Sampler

In reinforcement learning, there is a concept called Exploration-Exploitation Trade-off.

In the previous two steps, you took turns implementing the two extremes:

Exploration: exploring offers (random sampling)
Exploitation: selecting the best offerer (greedy sampling)
In the Exploration stage, you narrow down the uncertainty by trying out different options, adding to your knowledge of the offerers. In the Exploitation stage, you choose the best one based on your current knowledge (accumulated statistics). In what proportion should you follow this and that strategy? How to combine them?

![Alt text](/img/image.png)

### ε-greedy

Implement the simplest multi-armed bandit algorithm, called epsilon-greedy, which combines the best of both worlds. With probability 1-ε it will maximize Exploitation by choosing an offerer that maximizes revenue, and with probability ε it will choose a random one from among the offers.

![Alt text](/img/image-1.png)

## 4. Regret

Congratulations, you've just implemented your Reinforcement Learning algorithm!

Finally, the final step. What if we implement something even smarter than switching between the two extremes?

One way to evaluate how an RL algorithm will behave is to build a simulation. At each step, we know which "handle" yields the maximum revenue (in our case, which offer). However, our service doesn't, and the most interesting thing is the sequence in which it pulls these knobs and studies their behavior.

Knowing which "handle" is the best and how much (reward) it is likely (CR) to bring in, we can calculate the maximum possible amount of money we could make, knowing the state of the environment ahead of time. However, our algorithm is not an oracle, so it performs suboptimal actions and earns not the maximum amount of money.

The difference between the maximum possible payoff and the actual payoff is called regret.

The Epsilon-hungry algorithm is known for the fact that even in situations where the current "handles" are fully explored and give a predictable win, it still continues to explore a given % of clicks.

Good solution is UCB algorithm or Thompson sampling.

![Alt text](/img/image-2.png)

UCB-algorithm estimates not the average win, but the one with the maximum upper confidence bound. The confidence interval is wider the larger the uncertainty is (the less we try some handle).