# March Madness
Codebase for my submission to Kaggle's yearly March Machine Learning Mania competition.

## Site

Check out my website for this project. It expands on the explanations below, and it also has an interactive dashboard for viewing model insights: https://unclebrod.github.io/march-madness/

## Methodology

My current model is a probabilistic model, built using the [numpyro](https://num.pyro.ai/en/stable/) library. For each game, I calculate the number of possessions from the box score data using [Ken Pomeroy's methodology](https://kenpom.com/blog/the-possession/). The model estimates pace, offensive, and defensive ratings (random effects) for each team, controlling for the opponent.

Points per possession are a linear combination of a team's offensive rating, the opponent's defensive rating, and fixed effects (including features for rest, travel, and tournament indicators). Home court is also included as a random effect, allowing this value to be different for different teams. For example, Denver has a higher home court advantage given its altitude. 

Pace (here, possessions per minute) is derived as an average of each team's pace rating, plus fixed effects that are largely the same as above.

The model uses these ratings in order to estimate each team's score for a game. Possessions are modeled using a Poisson distribution; each team's score is modeled using a negative binomial which is similar to the Poisson except it has an extra parameter that allows the variance to be different than the mean.

Ratings are on a season-team level. They are modeled using a Gaussian walk, which allows a given season to serve as a prior for the next.

There is also some within-season weighting so that games at the end of the season count a bit more than games at the beginning.

The model data is on the game-team level, meaning there are two rows present for a given game where each team is represented once as the defensive team and once as the offensive.

The model is presently defined in `src/models/ppp.py`, where `ppp` stands for `points per possession`.

## Data Sources

I primarily use the data provided to us from the competition. The main outside source is Google's geocoding API which I use to get latitude, longitude, and elevation information for travel features in the model.
