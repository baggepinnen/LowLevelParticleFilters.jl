# How to tune a Kalman filter

The celebrated Kalman filter finds applications in many fields of engineering and economics. While many are familiar with the basic concepts of the Kalman filter, almost equally many find the "tuning parameters" associated with a Kalman filter non intuitive and difficult to tune. While there are several parameters that can be tuned in a real-world application of a Kalman filter, we will focus on the most important ones: the process and measurement noise covariance matrices.

## The Kalman filter
The Kalman filter is a form of Bayesian estimation algorithm that estimates the state of a linear dynamical system, subject to Gaussian noise acting on the measurements as well as the dynamics. More precisely, let the dynamics of a discrete-time linear dynamical system be given by
```math
\begin{aligned}
x_{k+1} &= Ax_k + Bu_k + w_k\\
y_k &= Cx_k + Du_k + e_k
\end{aligned}
```
where ``x_k \in \mathbb{R}^{n_x}`` is the state of the system at time ``k``, ``u_k \in \mathbb{R}^{n_u}`` is an external input, ``A`` and ``B`` are the state transition and input matrices respectively and ``w_k \sim N(0, R_1)`` and ``e_k \sim N(0, R_1)``, are normally distributed process noise and measurement noise terms respectively. A state estimator like the Kalman filter allows us to estimate ``x`` given only noisy measurements ``y \in \mathbb{R}^{n_y}``, i.e., without necessarily having measurements of all the components of ``x`` available.[^1] For this reason, state estimators are sometimes referred to as *virtual sensors*, i.e., they allows use to *estimate what we cannot measure*.

[^1]: Under a technical condition on the *observability* of the system dynamics.

The popularity of the Kalman filter is due to several reasons, for one, it is the *optimal estimator in the mean-square sense* if the system dynamics is linear (can be time varying) and the noise acting on the system is Gaussian. In most practical applications, neither of these conditions hold exactly, but they often hold sufficiently well for the Kalman filter to remain useful. A perhaps even more useful property of the Kalman filter is that the posterior probability distribution over the state remains Gaussian throughout the operation of the filter, making it efficient to compute and store. 

## What does "tuning the Kalman filter" mean?

To make use of a Kalman filter, we obviously need the dynamical model of the system given by the four matrices ``A, B, C`` and ``D``. We furthermore require a choice of the covariance matrices ``R_1`` and ``R_2``, and it is here a lot of aspiring Kalman filter users get stuck. The covariance matrix of the measurement noise is often rather straightforward to estimate, just collect some measurement data when the system is at rest and compute it's covariance, but we often lack any and all feeling for what the process noise covariance, ``R_1``, should be.

In this blog post, we will try to give some intuition for how to choose the process noise covariance matrix ``R_1``. We will come at this problem from a *disturbance-modeling* perspective, i.e., trying to reason about what disturbances act on the system and how, and what those imply for the structure and value of the covariance matrix ``R_1``. 
## Disturbance modeling

Intuitively, a disturbance acting on a dynamical system is some form of *unwanted* input. If you are trying to control the temperature in a room, it may be someone opening a window, or the sun shining on your roof. If you are trying to keep the rate of inflation at 2%, the disturbance may be a pandemic.

The linear dynamics assumed by the Kalman filter, here on discrete-time form
```math
x_{k+1} = Ax_k + Bu_k + w_k
```
makes it look like we have very little control over the shape and form of the disturbance ``w``, but there are a lof of possibilities hiding behind this equation.

In the equation above, the disturbance ``w`` has the same dimension as the state ``x``. Implying that the covariance matrix ``R_1 \in \mathbb{R}^{n_x \times n_x}`` has ``n_x^2`` parameters. We start by noting that for this to be a covariance matrix, it has to be symmetric and positive semi-definite. This means that only an upper or lower triangle of ``R_1`` contain free parameters. We further note that we can restrict the influence of ``w`` to a subset of the equations by introducing an input matrix ``B_w`` such that
```math
x_{k+1} = Ax_k + Bu_k + B_w \tilde{w}_k
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

[^2]: As a transfer function in the Laplace domain, the double integrator looks like ``P(s)/F(s) = \frac{1}{s^2}`` where ``P`` is the Laplace-transform of the position and ``F`` that of the force.

Now, what disturbances could possibly act on this system? The relation between velocity ``v`` and position ``p`` is certainly deterministic, and we cannot disturb the position of a system other than by continuously changing the velocity first (otherwise an infinite force would be required). This means that any disturbance acting on this system must take the form of a *disturbance force*, i.e., ``w = B_w w_f`` where ``B_w = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}``. A disturbance force ``w_f`` may be something like friction or air resistance etc. This means that the disturbance has a single degree of freedom only, and we can write the dynamics of the system as
```math
\dot x = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} f + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} w_f
```
where ``w_f`` is a scalar. This further means that the covariance matrix ``R_1`` has a *single free parameter only*, and we can write it as
```math
R_1 = \sigma_w^2 B_w B_w^{T} = \dfrac{\sigma_w^2}{m^2} \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
```
where ``\sigma_w^2`` is the variance of the disturbance ``w``. This is now our tuning parameter that we use to trade off the filter response time vs. the noise in the estimate.

What may initially have appeared as a tuning parameter ``R_1`` with three parameters to tune, has now been reduced to a single parameter by reasoning about how a possible disturbance acts on the system dynamics! The double integrator is a very simple example, but it illustrates the idea that the structure of the disturbance covariance matrix ``R_1`` is determined by the structure of the system dynamics and the form of the disturbance.

## But white noise, really?

Having had a look at the structural properties of the dynamics noise, let's now consider its *spectrum*. With noise like ``w_k \sim N(0, R_1)``, where ``w_k`` is uncorrelated with ``w_j`` for ``j \neq k``, is called *white noise* in analogy with white light, i.e., "containing all frequencies", or, "has a flat spectrum". White noise can often be a reasonable assumption for measurement noise, but much less so for dynamics noise. If we come back to the example of the temperature controlled room, the disturbance implied by the sun shining on the roof is likely dominated by low frequencies. The sun goes up in the morning and down in the evening, and clouds that may block the sun for a while do not move infinitely fast etc. For a disturbance like this, modeling it as white noise may not be the best choice.

Fear not, we can easily give color to our noise and still write the resulting model on the form 
```math
\dot x = Ax + Bu + B_w w
```


Let's say that our linear system ``P`` can be depicted in block-diagram form as follows:
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
Here, ``w`` is filtered through another linear system ``W`` to produce ``\tilde{w}``. If ``w`` has a flat white spectrum, the spectrum of ``\tilde{w}`` will be colored by the frequency response of ``W``. Thus, if we want to model that the system is affected by low-frequency noise ``w̃``, we can choose ``W`` as some form of low-pass filter. If we write ``W`` on statespace form as 
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
the augmented model has a state vector that is comprised of both the state vector of the original system ``P``, as well as the state vector ``x_w`` of the *disturbance model* ``W``. If we run a Kalman filter with this augmented model, the filter will estimate both the state of the original system ``P`` as well as the state of the disturbance model ``W`` for us!

It may at this point be instructive to reflect upon why we performed this additional step of modeling the disturbance? By including the disturbance model ``W``, we tell the Kalman filter what frequency-domain properties the disturbance has, and the filter can use these properties to make better predictions of the state of the system. This brings us to another key point of making use of a state estimator, it can perform *sensor fusion*.

## Sensor fusion
By making use of models of the dynamics, disturbances and measurement noises, the state estimator performs something often referred to as "sensor fusion". As the name suggests, sensor fusion is the process of combining information from multiple sensors to produce a more accurate estimate of the state of the system. In the case of the Kalman filter, the state estimator combines information from the dynamics model, the measurement model and the disturbance models to produce a more accurate estimate of the state of the system. We will contrast this approach to two common state-estimation heuristics
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

To learn the details on ZoH discretization consult [Discretization of linear state space models (wiki)](https://en.wikipedia.org/wiki/Discretization#discrete_function). Here, we will simply state a convenient way of computing this discretization, using the matrix exponential. Let ``A_c`` and ``B_c`` be the continuous-time dynamics and input matrices, respectively. Then, the discrete-time dynamics and input matrices are given by
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
\frac{T_s^4}{4} & \frac{T_s^3}{2} \\
\frac{T_s^3}{2} & T_s^2
\end{bmatrix}
```
This may look complicated, but it still has a single tuning parameter only, ``\sigma_w``.


## Putting it all together
We will now try to put the learnings from above together, applied to a slightly more complicated example. This time, we will consider a double-mass model, where two masses are connected by a spring and a damper, and an input force can be applied to one of the masses. The model, without disturbances, is given by
```math
\begin{aligned}
\dot x &= \begin{bmatrix}
0 & 1 & 0 & 0 \\
-k/m_1 & -c/m_1 & k/m_1 & c/m_1 \\
0 & 0 & 0 & 1 \\
k/m_2 & c/m_1 & -k/m_2 & -c/m_2 \end{bmatrix} x + \begin{bmatrix} 0 \\ 1/m_1 \\ 0 \\ 0 \end{bmatrix} u\\
y =& \begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix} x + e
\end{aligned}
```
where ``x = [p_1, v_1, p_2, v_2]`` is the state vector, ``u`` is the input force and ``y`` is the measured position of the first mass. The parameters ``m_1``, ``m_2``, ``k``, and ``c`` are the masses, spring constant, and damping constant, respectively.

What disturbances could act on such a system? One could imagine a friction force acting on the masses, indeed, most systems with moving parts are subject to friction. Friction is often modeled as a low-frequency disturbance, in particular Coulomb friction. The Coulomb friction is constant as long as the velocity does not cross zero, at which point it changes sign. We thus adopt an integrating model of this disturbance.

To model the fact that we are slightly uncertain about the dynamics of the flexible transmission, we could model a disturbance force acting on the spring. This could account for an uncertainty in a linear spring constant, but also model the fact that the spring is not perfectly linear, i.e., it might be a *stiffening spring* or contain some *backlash* etc. The question is, what frequency properties should we attribute to this disturbance? Backlash is typically a low-frequency disturbance, but uncertainties in the stiffness properties of the spring would likely affect higher frequencies as well. We thus let this disturbance have a flat-spectrum and omit an model of its frequency properties.

With the friction disturbance ``w_f`` and the spring disturbance ``w_s``, we can write the model as


```math
\begin{aligned}
\dot x &= A_a x_a + B_a u + B_{w_f} w_f + B_{w_s} w_s\\
y &= C_a x_a
\end{aligned}
```
where
```math
A_a = \begin{bmatrix} A & C_{w_f} \\ 0 & A_{w_f} \end{bmatrix},
\quad B_a = \begin{bmatrix} B \\ 0 \end{bmatrix},
\quad B_{w_f} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1/m_1 \end{bmatrix},
\quad B_{w_s} = \begin{bmatrix} 0 \\ 1/m_1 \\ 0 \\ -1/m_2 \\ 0 \end{bmatrix}
\quad C_a = \begin{bmatrix} C & 0 \end{bmatrix}
```

and ``x_a = \begin{bmatrix} x \\ x_{w_f} \end{bmatrix}``. When modeling ``w_f`` as an integrating disturbance ``W_f = 1/s^2``, we get the dynamics
```math
A_{w_f} = 0, \quad B_{w_f} = 1/m_1, \quad C_{w_f} = 1
```


### In code
To demonstrate how constructing a Kalman filter for the double-mass system would look using JuliaSim, we start by defining the dynamics in the form of a `StateSpace` model, as well as all the input matrices.

```@example DISTURBANCE_ESTIMATION
# using JuliaSimControl

using ControlSystemsBase, LowLevelParticleFilters, Plots, LinearAlgebra, Distributions, RobustAndOptimalControl

Ts = 0.01 # Sample time
m1 = 1    # Mass 1
m2 = 1    # Mass 2
k = 100   # Spring constant
c = 1     # Damping constant
A = [0 1 0 0;
     -k/m1 -c/m1 k/m1 c/m1;
     0 0 0 1;
     k/m2 c/m1 -k/m2 -c/m2]
B = [0; 1/m1; 0; 0]
C = [1 0 0 0]

Cwf = [0, 1, 0, 0]
Bwf = [0; 0; 0; 0; 1/m1]
Bws = [0; 1/m1; 0; -1/m2; 0]
Awf = 0
Aa = [A Cwf;
      zeros(1, 4) Awf]
Ba = [B; 0]
Ca = [C 0]
P  = ss(A, B, C, 0)    # Continuous-time system model
Pa = ss(Aa, Ba, Ca, 0) # Augmented system model
```

In practice, models are never perfect, we thus create another version of the model where the spring constant ``k`` is 10% larger. We will use this perturbed model to test how well the Kalman filter can estimate the state of the system in the presence of model mismatch.
```@example DISTURBANCE_ESTIMATION
k_actual = 1.1*k
A_actual = [0 1 0 0;
            -k_actual/m1 -c/m1 k_actual/m1 c/m1;
            0 0 0 1;
            k_actual/m2 c/m1 -k_actual/m2 -c/m2]
P_actual = ss(A_actual, B, C, 0) # Actual system model
nothing # hide
```

We then define the covariance matrices according to the above reasoning, and discretize them using the `c2d` function. Finally, we construct a `KalmanFilter` object.
```@example DISTURBANCE_ESTIMATION
σf  = 100  # Standard deviation of friction disturbance
σs  = 10  # Standard deviation of spring disturbance
σy  = 0.01 # Standard deviation of measurement noise
R1  = σf^2 * Bwf * Bwf' + σs^2 * Bws * Bws' # Covariance of continuous-time disturbance
R2  = σy^2 * I(1)     # Covariance of measurement noise

Pad = c2d(ss(Aa, [Ba Bwf Bws], Ca, 0), Ts) # Discrete-time augmented system model
Pd = Pad[:, 1]      # Discrete-time system model
Bwfd = Pad.B[:, 2]  # Discrete-time friction disturbance input matrix
Bwsd = Pad.B[:, 3]  # Discrete-time spring disturbance input matrix
R1d = σf^2 * Bwfd * Bwfd' + σs^2 * Bwsd * Bwsd' + 1e-8I |> Symmetric # Covariance of discrete-time disturbance

kf  = KalmanFilter(Pd, R1d, R2)
nothing # hide
```
When running a Kalman filter in a real-world application, we often receive and process measurements in real time. However, for the purposes of this blog post, we generate some inputs and outputs by simulating the system. We construct a PID controller `Cfb` and close the loop around this controller to to simulate how the system behave under feedback. We let the reference position be a random square signal, which we obtain by low-pass filtering random noise. To obtain both control input and measured output from the simulation, we form the feedback interconnection using the function `feedback_control`. 
```@example DISTURBANCE_ESTIMATION
Cfb = pid(100, 1, 0.2, Tf=2Ts)
G = feedback_control(P_actual, Cfb)*tf(1, [0.1, 1])
r = 0.1cumsum(sign.(sin.(0.1.*(0:0.02:10).^2)))
timevec = range(0, length=length(r), step=Ts)
res = lsim(G, r', timevec)
inputs = res.y[2,:]
outputs = res.y[1,:]
plot(res, ylabel=["Position mass 1" "Input signal"])
```

To simulate a Coulomb friction disturbance, we extract the velocity of the first mass, ``v_1 = x_2``, and compute the friction force as ``- k_f \operatorname{sign}(v)``. We then apply this disturbance to the input as seen by the Kalman filter in order to simulate the effect of the disturbance on the estimator. We then call `forward_trajectory` to perform Kalman filtering along the entire recorded trajectory, and plot the result alongside the true state trajectory as obtained by the function `lsim`.

```@example DISTURBANCE_ESTIMATION
wf = map(eachindex(res.t)) do i
    ui = inputs[i]
    v = res.x[2,i]
    -20*sign(v) # Coulomb friction disturbance
end

u = map(eachindex(res.t)) do i
    ui = inputs[i] + wf[i] # The input seen by the state estimator is the true input + the disturbance
    [ui]
end
y = [[yi + σy*randn()] for yi in outputs] # Outputs with added measurement noise
sol = forward_trajectory(kf, u, y)
true_states = lsim(P_actual, inputs', timevec).x
plot(sol)
plot!(true_states', sp=(1:4)', lab="True")
plot!(-wf, sp=5, lab="Friction force")
```
This figure shows the 5 state variables ``x_a = [p_1, v_1, p_2, v_2, x_{w_f}]`` of the augmented system model ``P_a``, as well as the input signal ``u`` and the measured output ``y``. In the plot of the estimated disturbance state ``x_5 = x_{w_f}``, we plot also the true friction force from the simulation. Analyzing the results, we see that the state estimator needs a couple of samples to catch up to the change in the friction disturbance, but it does correctly converge to a noisy estimate of the friction force in steady state. 


## Estimating covariance from data
In the example above, we treated ``\sigma_f^2`` and ``\sigma_s^2`` as tuning parameters to be manually selected. However, it is possible to use data to estimate these parameters using, e.g., maximum likelihood estimation. This blog post is long enough as it is, so that's a story for another time.[^3]

[^3]: Here's part of that story if you are really curious. https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/parameter_estimation/

## Concluding remarks

In this blog post, we have discussed how we could give meaning to the all-so-important covariance matrix of the dynamics noise appearing in a Kalman filter. By reasoning about what disturbances we expect to act on the system, both due to disturbance inputs and due to model mismatch, we can construct a covariance matrix that is consistent with our expectations. Here, we modeled disturbances in order to improve the estimation of the state of the system, however, in some applications the disturbance estimate itself is the primary quantity of interest. Examples of disturbance estimation include estimation of external contact forces acting on a robot, and room occupancy estimation (a person contributes approximately 100W of heat power to their surrounding). With all these examples in mind, no wonder people sometimes call state estimators for "virtual sensors"!

### Potential improvements
Although we managed to estimate the friction force acting on the double-mass system above, there is still plenty of room for improvement. If we have an accurate, nonlinear model of the friction, we could likely cancel out the friction disturbance much faster than what our simple integrating model did.[^4] Even if we did not have an accurate model, making use of the fact that the friction force has a complicated behavior when the velocity crosses zero would likely improve the performance of the estimator. This could easily be done in a Kalman-filtering framework by giving the covariance a little bump when the velocity crosses or approaches zero. An example of such an adaptive covariance approach is given in the [Adaptive Kalman-filter tutorial](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/adaptive_kalmanfilter/).

[^4]: Using a Kalman filter with an integrating disturbance model in a state-feedback controller results in a controller with integral action. See [Integral action](https://help.juliahub.com/juliasimcontrol/dev/integral_action/) for more details.

### Further reading
To learn more about state estimation in JuliaSim, see the following resources:
- [State estimation with ModelingToolkit](https://help.juliahub.com/juliasimcontrol/dev/examples/state_estimation/)
- [Noise adaptive Kalman filter tutorial](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/adaptive_kalmanfilter/)
- [Parameter estimation for state estimators](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/parameter_estimation/)
- [Estimation of Kalman filters directly from data using statespace model estimation](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/ss/)
- [Disturbance modeling with ModelingToolkit](https://help.juliahub.com/juliasimcontrol/dev/examples/mtk_disturbance_modeling/)
- [Disturbance modeling for Model-Predictive Control (MPC)](https://help.juliahub.com/juliasimcontrol/dev/examples/disturbance_rejection_mpc/)
