# place_data_sci_proj
Data science project for placement exam

## Deadliest natural disasters plots

The data for the following plots of the deadliest natural disasters each year from 1901-2026 were scraped from the Wikipedia page: [List of natural disasters by death toll](https://en.wikipedia.org/wiki/List_of_natural_disasters_by_death_toll#).

![Figure 1](/figures/fig1.png)

This graph shows the deadliest natural disaster each year from 1901-2026, plotted by the death toll for each disaster, excluding epidemics and famines. The color of each point represents the type of disaster that occurred, such as flood or landslide. For certain disasters, only an estimated range was available for the death toll; in these cases, the plot shows the midpoint of the range as the death toll. In 1972, there are two disasters represented on the graph, because it's unclear which was the deadliest that year: the Qir earthquake which caused 5,374 deaths and the Managua earthquake, which caused 4,000-11,000 deaths.

This graph highlights how in unpredictable years, certain events caused extremely high numbers of deaths, appearing as labelled outliers in the graph. The median number of deaths caused by the deadliest disaster in a given year is 6,434. The four events with the highest death tolls are much higher: the 1931 China floods (~2.2 million deaths), 1976 Tangshan earthquake (~450,000 deaths), 1970 Bhola cyclone (~400,000 deaths), and 1920 Haiyuan earthquake (~270,000 deaths).

![Figure 2](/figures/fig2.png)

This graph shows the same data as the previous one, but this time plotting the number of deaths for each event on a log scale. This helps to better visualize the spread of the data, rather than highlighting the highest death toll events. There doesn't appear to be a strong trend in death toll over time, meaning the death toll from the deadliest natural disaster each year hasn't been increasing or decreasing over time. You can also see from the graph that earthquake is the most common type of deadliest natural disaster in this time period, with tropical cyclones as the second most common type. Out of the 126 years represented on the graph, in 61 years the deadliest disaster involved an earthquake, and in 37 years the deadliest disaster involved a tropical cyclone.

## Gradient descent testing

### Background and method

In this section, we test the dependence of the gradient descent algorithm on the algorithm's step size. We are minimizing the loss function $L(b) = ||y-bx||^2$, where $y$ and $x$ are vectors and $b$ is a scalar, over $b$. The gradient descent algorithm iterates on an initial guess of $b$, $b_0$, such that $b_{n+1} = b_n - eL'(b)$, where $e$ is the step size and $L'(b) = -2(y-bx) \cdot x$.

To test the dependence of this algorithm on $e$, we generated 100 random problem sets of unique $x$, $y$, and $b_0$ values. We tested vectors of length 2 for $x$ and $y$ for simplicity. The elements of $x$, $y$, and $_b0$ were sampled randomly from the $N(0, 1)$ distribution, to avoid extreme values that cause the gradient $L'(b)$ to be extremely large. We then ran gradient descent for each of the 100 problem sets, once for each a range of values of $e$: 1e-5, 1e-4, 0.001, 0.01, 0.1, 0.5, and 1. There are several possible stopping criteria for gradient descent, including relative and absolute criteria on the change in b, change in L(b), and L'(b). For the purposes of this testing, we considered gradient descent to have converged when the relative change L(b) was less than a tolerance of 1e-6. We set a maximum number of iteration of 1e6; if the algorithm did not reach the stopping criteria before then, it was considered not converged.

