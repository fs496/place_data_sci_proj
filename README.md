# place_data_sci_proj
Data science project for placement exam

## Deadliest natural disasters plots

The data for the following plots of the deadliest natural disasters each year from 1901-2026 were scraped from the Wikipedia page: [List of natural disasters by death toll](https://en.wikipedia.org/wiki/List_of_natural_disasters_by_death_toll#).

**Figure 1**

![Figure 1](/figures/fig1.png)


This graph shows the deadliest natural disaster each year from 1901-2026, plotted by the death toll for each disaster, excluding epidemics and famines. The color of each point represents the type of disaster that occurred, such as flood or landslide. For certain disasters, only an estimated range was available for the death toll; in these cases, the plot shows the midpoint of the range as the death toll. In 1972, there are two disasters represented on the graph, because it's unclear which was the deadliest that year: the Qir earthquake which caused 5,374 deaths and the Managua earthquake, which caused 4,000-11,000 deaths.

This graph highlights how in unpredictable years, certain events caused extremely high numbers of deaths, appearing as labelled outliers in the graph. The median number of deaths caused by the deadliest disaster in a given year is 6,434. The four events with the highest death tolls are much higher: the 1931 China floods (~2.2 million deaths), 1976 Tangshan earthquake (~450,000 deaths), 1970 Bhola cyclone (~400,000 deaths), and 1920 Haiyuan earthquake (~270,000 deaths).

**Figure 2**

![Figure 2](/figures/fig2.png)

This graph shows the same data as the previous one, but this time plotting the number of deaths for each event on a log scale. This helps to better visualize the spread of the data, rather than highlighting the highest death toll events. There doesn't appear to be a strong trend in death toll over time, meaning the death toll from the deadliest natural disaster each year hasn't been increasing or decreasing over time. You can also see from the graph that earthquake is the most common type of deadliest natural disaster in this time period, with tropical cyclones as the second most common type. Out of the 126 years represented on the graph, in 61 years the deadliest disaster involved an earthquake, and in 37 years the deadliest disaster involved a tropical cyclone.

## Gradient descent testing

### Background and method

In this section, we test the dependence of the gradient descent algorithm on the algorithm's step size. We are minimizing the loss function $L(b) = ||y-bx||^2$, where $y$ and $x$ are vectors and $b$ is a scalar, over $b$. The gradient descent algorithm iterates on an initial guess of $b$, $b_0$, such that $b_{n+1} = b_n - eL'(b)$, where $e$ is the step size and $L'(b) = -2(y-bx) \cdot x$.

To test the dependence of this algorithm on $e$, we generated 100 random problem sets of unique $x$, $y$, and $b_0$ values. We tested vectors of length 2 for $x$ and $y$ for simplicity. The elements of $x$, $y$, and $b_0$ were sampled randomly from the $N(0, 1)$ distribution, to avoid extreme values that cause the gradient $L'(b)$ to be extremely large. We then ran gradient descent for each of the 100 problem sets, once for each of a range of $e$ values: 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, and 1. There are several possible stopping criteria for gradient descent, including on the absolute and relative change in $b$, absolute and relative change in $L(b)$, and magnitude of $L'(b)$. For the purposes of this testing, we considered gradient descent to have converged when the relative change in $L(b)$ was less than a tolerance of $10^{-6}$. We set a maximum number of iterations of $10^{6}$; if the algorithm did not reach the stopping criteria before then, it was considered not converged.

### Results and analysis

**Figure 3**

![Figure 3](/figures/fig_conv.png)

**Figure 4**

![Figure 4](/figures/fig_steps.png)

**Figure 5**

![Figure 5](/figures/fig_err.png)

Figure 3 shows the convergence rate (percent of runs that successfully converged) for each value of $e$. We can see that the algorithm's convergence rate is at or close to 100% for step sizes 0.1 and smaller, but drops significantly for step sizes 0.5 and 1. This is a consequence of the fact that if $e$ is too large, gradient descent can easily overshoot the true value of $b_{min}$, leading to divergence. Figure 4 shows the distribution of the number of steps needed to converge (for runs that successfully converged) over values of $e$. We can see that while all values of $e$ less than 0.5 have high convergence rates, decreasing $e$ by an order of magnitude results in a roughly order of magnitude increase in the number of steps needed to converge. This corresponds to the fact that $|b_{n+1} - b_n|$ is constrained by the size of $e$; if $e$ is extremely small, the algorithm will require many more steps to converge. Figure 5 shows a final performance metric, the percent error in the estimated value of $b$ compared to its true value (accuracy) for runs that converged. Decreasing $e$ significantly decreases the accuracy of the estimate; for example, at $e=10^{-5}$, the median percent error is around 30%. This corresponds to the fact that for extremely small $e$, it is possible for the algorithm to reach the stopping criteria (AKA convergence) despite not actually having reached the minimum of the loss function, simply because the step size itself is making the change in loss, $|L_{n+1} - L_{n}|$, artificially small.

Thus, from a performance perspective, the optimal value of $e$ for the problems tested lies in the 0.01-0.1 range. In general, the best value of $e$ must balance the need to take bigger steps for better speed and accuracy, with the need to take smaller steps to avoid overshooting.

We also examined the relationship between the performance of the algorithm (convergence rate, number of steps, and accuracy) with other factors, such as $||x||$, $||y||$, $\frac{||x||}{||y||}$, the angle between $x$ and $y$ ($\theta$), and the percent error in the initial guess $b_0$ compared to the true $b$ (how far off the initial guess is). Only $||x||$ showed a noticeable relationship to performance.

**Figure 6**

![Figure 6](/figures/Convergence%20rate%20(percent)%20by%20Step%20size%20(e)%20and%20Length%20of%20x.png)

Figure 6 shows convergence rates by $||x||$ as well as $e$. A noticeable trend is that even when $e$ is large, for small $||x||$ convergence rates are close to 100%. This is likely because the small magnitude of $x$ decreases the change in $b$ at each iteration, making overshooting less likely.

**Figure 7**

![Figure 7](/figures/Median%20number%20of%20steps%20per%20run%20by%20Step%20size%20(e)%20and%20Length%20of%20x.png)

Figure 7 shows the median number of steps needed for convergence by $||x||$ as well as $e$. A noticeable trend is that when both $e$ and $||x||$ are small, the number of steps needed for convergence increases dramatically (up to more than 250,000 steps). This is likely because the small magnitude of $x$ decreases the change in $b$ at each iteration, making the algorithm much slower. Thus, small $||x||$ improves convergence for large $e$ but decreases speed for small $e$.
