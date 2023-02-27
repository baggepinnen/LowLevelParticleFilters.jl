# How to tune a Kalman filter

The celebrated Kalman filter finds applications in many fields of engineering and economics. While many are familiar with the basic concepts of the Kalman filter, almost equally many find the "tuning parameters" associated with a Kalman filter non intuitive and difficult to tune. While there are several parameters that can be tuned in a real-world application of a Kalman filter, we will focus on the most important ones: the process and measurement noise covariance matrices.

## The Kalman filter
The Kalman filter is a form of Bayesian estimation algorithm that estimates the state of a linear dynamical system, subject to Gaussian noise acting on the measurements as well as the dynamics. More precisely, let the dynamics of a linear dynamical system be given by
```math
\begin{aligned}
x(k+1) &= Ax(k) + Bu(k) + w(k)\\
y(k) &= Cx(k) + Du(k) + e(k)
\end{aligned}
```
where ``x(k) \in \mathbb{R}^{n_x}`` is the state of the system at time ``k``, ``u(k) \in \mathbb{R}^{n_u}`` is an external input, ``A`` and ``B`` are the state transition and input matrices respectively and ``w(k) \sim N(0, R_1)`` and ``e(t) \sim N(0, R_1)``, are normally distributed process noise and measurement noise terms respectively. A state estimator like the Kalman filter allows us to estimate ``x`` given only noisy measurements ``y \in \mathbb{R}^{n_y}``, i.e., without necessarily having measurements of all the components of ``x`` available.[^1] For this reason, state estimators are sometimes referred to as *virtual sensors*, i.e., they allows use to *estimate what we cannot measure*.

[^1]: Under a technical condition on the *observability* of the system dynamics.

The popularity of the Kalman filter is due to several reasons, for one, it is the *optimal estimator in the mean-square sense* if the system dynamics is linear (can be time varying) and the noise acting on the system is Gaussian. In most practical applications, neither of these conditions hold exactly, but they often hold sufficiently well for the Kalman filter to remain useful. A perhaps even more useful property of the Kalman filter is that the posterior probability distribution over the state remains Gaussian throughout the operation of the filter, making it efficient to compute and store. 

## What does "tuning the Kalman filter" mean?

To make use of a Kalman filter, we obviously need the dynamical model of the system given by the four matrices ``A, B, C`` and ``D``. We furthermore require a choice of the covariance matrices ``R_1`` and ``R_2``, and it is here a lot of aspiring Kalman filter users get stuck. The covariance matrix of the measurement noise is often rather straightforward to estimate, just collect some measurement data when the system is at rest and compute it's covariance, but we often lack any and all feeling for what the process noise covariance, ``R_1``, should be.

In this blog post, we will try to give some intuition for how to choose the process noise covariance matrix ``R_1``. We will come at this problem from a *disturbance-modeling* perspective, i.e., trying to reason about what disturbances act on the system and how, and what those imply for the structure and value of the covariance matrix ``R_1``. We will also demonstrate how the covariance matrix ``R_1`` can be estimated from data using maximum-likelihood estimation.

## Disturbance modeling

Intuitively, a disturbance acting on a dynamical system is some form of *unwanted* input. If you are trying to control the temperature in a room, it may be someone opening a window, or the sun shining on your roof. If you are trying to keep the rate of inflation at 2%, the disturbance may be a pandemic.

The linear dynamics assumed by the Kalman filter, here on discrete-time form
```math
x(k+1) = Ax(k) + Bu(k) + w(k)
```
makes it look like we have very little control over the shape and form of the disturbance ``w``, but there are a lof of possibilities hiding behind this equation.

In the equation above, the disturbance ``w`` has the same dimension as the state ``x``. Implying that the covariance matrix ``R_1 \in \mathbb{R}^{n_x \times n_x}`` has ``n_x^2`` parameters. We start by noting that for this to be a covariance matrix, it has to be symmetric and positive semi-definite. This means that only an upper or lower triangle of ``R_1`` contain free parameters. We further note that we can restrict the influence of ``w`` to a subset of the equations by introducing a input matrix ``B_w`` such that
```math
x(k+1) = Ax(k) + Bu(k) + B_w \tilde{w}(k)
```
where ``w = B_w \tilde{w}`` and ``\tilde{w}`` may have a smaller dimension than ``w``. To give a feeling for why this might be relevant, we consider a very basic system, the *double integrator*.

A double integrator appears whenever Newton's second law appears
```math
f = ma
```
This law states that the acceleration ``a`` of a system is proportional to the force ``f`` acting on it, on continuous-time statespace form,[^2] this takes the form
```math
\begin{aligned}
\dot p(t) &= v(t)\\
m\dot v(t) &= f(t)
\end{aligned}
```
or on the familiar matrix form:
```math
\dot x = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} f 
```
where ``x = [p, v]^T``. 

[^2]: As a transfer function in the Laplace domain, this system looks like ``P(s) = \frac{1}{s^2}F(s)`` where ``P`` is the Laplace-transform of the position and ``F`` that of the force.

Now, what disturbances could possibly act on this system? The relation between velocity ``v`` and position ``p`` is certainly deterministic, and we cannot disturb the position of a system other than by continuously changing the velocity first (otherwise an infinite force would be required). This means that any disturbance acting on this system must take the form of a *disturbance force*, i.e., ``\tilde{w} = B_w w`` where ``B_w = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}``. A disturbance force may be something like friction or air resistance etc. This means that the disturbance has a single degree of freedom only, and we can write the dynamics of the system as
```math
\dot x = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} f + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} w
```
where ``w`` is a scalar (we abuse the notation slightly and use ``w`` rather than ``\tilde{w}`` for simplicity). This further means that the covariance matrix ``R_1`` has a *single free parameter only*, and we can write it as
```math
R_1 = \sigma_w^2 B_w B_w^{T} = \begin{bmatrix} 0 & 0 \\ 0 & \frac{\sigma_w^2}{m^2} \end{bmatrix}
```
where ``\sigma_w^2`` is the variance of the disturbance ``w``. This is now our tuning parameter that we use to trade off the filter response time vs. the noise in the estimate.

What may initially have appeared as a tuning parameter ``R_1`` with three parameters to tune, has now been reduced to a single parameter by reasoning about how a possible disturbance acts on the system dynamics! The double integrator is a very simple example, but it illustrates the idea that the structure of the disturbance covariance matrix ``R_1`` is determined by the structure of the system dynamics and the form of the disturbance.

## But white noise, really?

Having had a look at the structural properties of the dynamics noise, let's now consider its *spectrum*. With noise like ``w(k) \sim N(0, R_1)``, where ``w(k)`` is uncorrelated with ``w(j)`` for ``j \neq k`` is called *white noise* in analogy with white light, i.e., "containing all frequencies", or, "has a flat spectrum". White noise can often be a reasonable assumption for measurement noise, but much less so for dynamics noise. If we come back to the example of the temperature controlled room, the disturbance implied by the sun shining on the roof is likely dominated by low frequencies. The sun goes up in the morning and down in the evening, and clouds may block the sun for a while do not move infinitely fast etc. For a disturbance like this, modeling it as white noise may not be the best choice.

Fear not, we can easily give color to our noise and still write the resulting model on the form 
```math
\dot x = Ax + Bu + B_w w
```


Let's say that our linear system ``P`` is can be depicted in block-diagram form as follows:
```
   │w
   ▼
┌─────┐
│  W  │
└──┬──┘
   │w̃
   │  ┌─────┐
 u ▼  │     │ y
 ──+─►│  P  ├───►
      │     │
      └─────┘
```
Here, ``\tilde{w}`` is filtered through another linear system ``W`` to produce ``w̃``. If ``w`` has a flat white spectrum, the spectrum of ``w̃`` will be colored by the frequency response of ``W``. Thus, if we want to model that the system is affected by low-frequency noise ``w̃``, we can choose ``W`` as some form of low-pass filter. If we write ``W`` on statespace form as 
```math
\begin{aligned}
\dot x_w &= A_w x_w + B_w w \\
w̃ &= C_w x_w
\end{aligned}
```
we can form an *augmented system model* ``P_a`` as follows:
```math
\begin{aligned}
\dot x &= A_a x_a + B_a u + B_{aw} w \\
y &= C_a x_a
\end{aligned}
```
where
```math
A_a = \begin{bmatrix} A & C_w \\ 0 & A_w \end{bmatrix}, \quad B_a = \begin{bmatrix} B \\ 0 \end{bmatrix}, \quad B_{aw} = \begin{bmatrix} 0 \\ B_w \end{bmatrix}, \quad C_a = \begin{bmatrix} C & 0 \end{bmatrix}
```
and
```math
x_a = \begin{bmatrix} x \\ x_w \end{bmatrix}
```
the augmented model has a state vector that is comprised of both the state vector of the original system ``P``, as well as the state vector ``x_w`` of the ``disturbance model`` ``W``. If we run a Kalman filter with this augmented model, the filter will estimate both the state of the original system ``P`` as well as the state of the disturbance model ``W`` for us!

It may at this point be instructive to reflect upon why we performed this additional step of modeling the disturbance? By including the disturbance model ``W``, we tell the Kalman filter what frequency-domain properties ``w`` has, and the filter can use these properties to make better predictions of the state of the system. This brings us to another key point of making use of a state estimator.

## Sensor fusion
By making use of models of the dynamics, disturbances and measurement noises, the state estimator performs something often referred to as *sensor fusion*. As the name suggests, sensor fusion is the process of combining information from multiple sensors to produce a more accurate estimate of the state of the system. In the case of the Kalman filter, the state estimator combines information from the dynamics model, the measurement model and the disturbance models to produce a more accurate estimate of the state of the system. We will contrast this approach to two common state-estimation heuristics
- Differentiation for velocity estimation
- Complementary filtering for orientation estimation

Velocity is notoriously difficult to measure in most applications, but measuring position is often easy. A naive approach to estimating velocity is to differentiate the position measurements. However, there are a couple of problems associated with this approach. First, the noise in the position measurements will be amplified by the differentiation, second, only differentiating the measured position ignores any information of the input to the system. Intuitively, if we know how the system behaves in response to an input, we should be able to use this knowledge to form a better estimate of both the position and the velocity?

Indeed, a Kalman filter allows you to estimate the velocity, taking into account both the input to the system and the noisy position measurement. If the model of the system is perfect, we do not even need a measurement, the model and the input is sufficient to compute what the velocity will be. In practice, models are never perfect, and we thus make use of the "fusion aspect" of the state estimator to incorporate the two different sources of information, the model and the measurement, to produce a better estimate of the velocity.

A slightly more complicated example is the complimentary filter. This filter is often used with inertial measurement units (IMUs), containing accelerometers and gyroscopes to estimate the orientation of a system. Accelerometers are often very noisy, but measure the correct orientation on average. Gyroscopes are often very accurate, but drift over time. The drift over time can be modeled as a low-frequency disturbance acting on the measurement. The complimentary filter combines the information from the accelerometer and the gyroscope to produce a more accurate estimate of the orientation. This is done by low-pass filtering the accelerometer measurement to get rid of the high-frequency measurement noise, and high-pass filtering the gyroscope measurement to get rid of the low-frequency drift.

We can arrive at a filter with these same properties, together with a dynamical model that indicates the system's response to control inputs, by using a Kalman filter. In this case, we would include two different disturbance models, acting on the accelerometer output ``y_a`` and the gyroscope output ``y_g`` like this
```
            │wa
            ▼
         ┌─────┐
         │  Wa │
         └──┬──┘
            │
    ┌─────┐ ▼
u   │     ├─+─► ya
───►│  P  │
    │     ├─+─► yg
    └─────┘ ▲
            │
            │
         ┌──┴──┐
         │  Wg │
         └─────┘
            ▲
            │wg
```
``W_a`` would here be chosen as some form of a low-pass filter, while ``W_g`` would be chosen as a high-pass filter. The complimentary filter makes the "complimentary assumption" ``W_g = 1 - W_a``, i.e., ``W_a`` and ``W_g`` sum to one. This is a simple and often effective heuristic, but the filter does not make any use of the input signal ``u`` to form the estimate, and will thus suffer from phase loss, sometimes called *lag*, in response to inputs. This is particularly problematic when there are communication *delays* present between the sensor and the state estimator. During the delay time, the sensor measurements contain no information at all about any system response to inputs. 


## Discretization
So far, I have switched between writing dynamics in continuous time, i.e., on the form
```math
\dot x(t) = A x(t) + B u(t)
```
and in discrete time
```math
x_{k+1} = A x_k + B u_k
```
Physical systems are often best modeled in continuous time, while some systems, notably those living inside a computer, are inherently discrete time. Kalman filters are thus most often implemented in discrete time, and any continuous-time model must be discretized before it can be used in a Kalman filter. For control purposes, models are often discretized using a zero-order-hold assumption, i.e., input signals are assumed to be constant between sample intervals. This is often a valid assumption for control inputs, but not always for disturbance inputs. If the sample rate is fast in relation to the time constants of the system, the discretization method used does not matter all too much. For the purposes of this tutorial, we will use the zero-order-hold (ZoH) assumption for all inputs, including disturbances.

To learn the details on ZoH discretization consult [Discretization of linear state space models (wiki)](https://en.wikipedia.org/wiki/Discretization#discrete_function). Here, we will simply state a convenient way of computing this discretization. Let ``A_c`` and ``B_c`` be the continuous-time dynamics and input matrices, respectively. Then, the discrete-time dynamics and input matrices are given by
```math
\begin{bmatrix}
A_d & B_d \\
0 & I
\end{bmatrix}
=
\exp\left(\begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} T_s\right)
```
where ``A_d`` and ``B_d`` are the discrete-time dynamics and input matrices, respectively, and ``T_s`` is the sample interval. The ``I`` in the bottom right corner is the identity matrix. To discretize the input matrix for a disturbance model, we simply replace ``B`` with ``B_w``, or put all the ``B`` matrices together by horizontal concatenation and discretize them all at once.

Discretizing the continuous time model of the double integrator with a force disturbance, we get
```math
\begin{aligned}
x_{k+1} &= \begin{bmatrix} 1 & T_s \\ 0 & 1 \end{bmatrix} x_k + \dfrac{1}{m}\begin{bmatrix} T_s^2/2 \\ T_s \end{bmatrix} f + \dfrac{1}{m}\begin{bmatrix} T_s^2/2 \\ T_s \end{bmatrix} w \\
y_k &= \begin{bmatrix} 1 & 0 \end{bmatrix} x_k + e_k
\end{aligned}
```
with the corresponding covariance matrix
```math
R_1 = \dfrac{\sigma^2}{m^2} \begin{bmatrix}
T_s^3/3 & T_s^2/2 \\
T_s^2/2 & T_s
\end{bmatrix}
```
This may look complicated, but it still has a single tuning parameter only, ``\sigma_w``.


## Putting it all together
Double-mass model

## Estimating covariance from data